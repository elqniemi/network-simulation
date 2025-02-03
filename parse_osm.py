#!/usr/bin/env python3
"""
This script parses an OSM PBF file and extracts a road network graph.
It works in two passes:

1. First, we count the number of distinct ways that reference each OSM node.
   (We use these counts later to decide where a road should be split.)
   A node is considered “important” (i.e. an intersection or endpoint) if it:
     - is the first or last node of a way, or
     - appears in more than one way.
     
2. Second, we load node coordinates and process each allowed way:
   - The way’s node list is “split” into segments at the important nodes.
   - Each segment gets its own (new) edge id and its geometry is stored as a LineString.
   - A new column “oneway” is added (True if the original way is one‑way,
     False if it is bidirectional).
   - Each segment’s endpoints become nodes in the final graph; they are recorded
     (with new node ids) in a node table.
     
The final output is a GeoPackage file with two layers:
  • “edges” – the network segments (with new edge ids, source/target, highway, base_speed, priority, oneway, etc.)
  • “nodes” – the intersections/endpoints (with new node ids, and the original OSM node id for reference)

In addition, after reprojecting from EPSG:4326 to EPSG:28992 (a meter‐based CRS),
a “length_m” column (in meters) and a “cost” column (travel time in seconds) are added.

Finally, disconnected areas (components) with fewer than 50 edges are removed.
"""

import osmium
from shapely.geometry import LineString, Point
import geopandas as gpd
import re
import argparse
import networkx as nx

# -------------------------------------------------------------------------
# Default Dutch speed limits (in km/h) for various highway types.
default_dutch_speeds = {
    "motorway":         100,
    "motorway_link":     50,
    "trunk":            100,
    "trunk_link":        50,
    "primary":           70,
    "primary_link":      50,
    "secondary":         500,
    "secondary_link":    50,
    "tertiary":          30,
    "tertiary_link":     30,
    "residential":       20,
    "living_street":     10,
    "service":           30,
    "unclassified":      30,
}

default_priority = {
    "motorway":         1,
    "motorway_link":    1,
    "trunk":            1.2,
    "trunk_link":       1.2,
    "primary":          1.2,
    "primary_link":     1.2,
    "secondary":        1.3,
    "secondary_link":   1.3,
    "tertiary":         1.5,
    "tertiary_link":    1.5,
    "residential":      1.8,
    "living_street":    3,
    "service":          10,
    "unclassified":     3,
}

# Set of highway tags that we consider suitable for car traffic.
allowed_highways = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential", "living_street",
    "service", "unclassified"
}

# -------------------------------------------------------------------------
# Helper function to parse a maxspeed string.
def parse_maxspeed(maxspeed_str, highway_type):
    """
    Parses the maxspeed tag value and returns a float (km/h).
    If parsing fails, returns the default speed from default_dutch_speeds.
    """
    try:
        s = maxspeed_str.strip().lower()
        if "mph" in s:
            m = re.search(r"(\d+)", s)
            if m:
                return float(m.group(1)) * 1.60934
        else:
            m = re.search(r"(\d+)", s)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return default_dutch_speeds.get(highway_type, 50)

# -------------------------------------------------------------------------
# First pass: Count how many distinct ways reference each node.
class CountHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        # For each node id, count the number of ways in which it appears.
        self.node_usage = {}

    def way(self, w):
        # Only process ways with a highway tag that are allowed.
        if 'highway' not in w.tags:
            return
        highway = w.tags.get('highway')
        if highway not in allowed_highways:
            return
        # Skip areas.
        if w.tags.get("area") == "yes":
            return
        if len(w.nodes) < 2:
            return

        # Use a set so that if a node appears twice in one way we only count it once.
        unique_nodes = set(n.ref for n in w.nodes)
        for node_id in unique_nodes:
            self.node_usage[node_id] = self.node_usage.get(node_id, 0) + 1

# -------------------------------------------------------------------------
# Second pass: Process nodes and ways; split ways into segments at intersections.
class RoadSegmentHandler(osmium.SimpleHandler):
    def __init__(self, node_usage):
        super().__init__()
        # Provided node usage dictionary from CountHandler.
        self.node_usage = node_usage
        # Dictionary mapping OSM node id to coordinate tuple (lon, lat).
        self.nodes = {}
        # List to hold the new edge (segment) records.
        self.edges = []
        # Mapping of OSM node ids that become graph nodes to new node ids.
        self.new_node_mapping = {}
        # List to hold node features (for the node table).
        self.node_features = []
        # Counter for new edge ids.
        self.edge_id_counter = 1

    def node(self, n):
        if n.location.valid():
            self.nodes[n.id] = (n.location.lon, n.location.lat)

    def way(self, w):
        # Only process ways with a highway tag.
        if 'highway' not in w.tags:
            return
        highway = w.tags.get('highway')
        if highway not in allowed_highways:
            return
        if w.tags.get("area") == "yes":
            return
        if len(w.nodes) < 2:
            return

        # Build lists of coordinates and OSM node ids.
        coords = []
        node_ids = []
        for n in w.nodes:
            if n.ref in self.nodes:
                coords.append(self.nodes[n.ref])
                node_ids.append(n.ref)
            else:
                # If any node location is missing, skip the way.
                return

        # Determine base speed.
        if 'maxspeed' in w.tags:
            base_speed = parse_maxspeed(w.tags.get('maxspeed'), highway)
        else:
            base_speed = default_dutch_speeds.get(highway, 50)
        
        # Determine priority.
        priority = default_priority.get(highway, 1)

        # Determine oneway information.
        # (OSM oneway tag values can be "yes", "1", "true", or "-1".)
        oneway_tag = w.tags.get("oneway", "no").lower()
        if oneway_tag in ["yes", "true", "1"]:
            oneway = 1
            reverse = 0
        elif oneway_tag == "-1":
            oneway = 1
            reverse = 1
        else:
            oneway = 0
            reverse = 0

        # For one-way roads with -1, reverse the order.
        if oneway and reverse:
            node_ids = list(reversed(node_ids))
            coords = list(reversed(coords))

        # Determine splitting indices.
        # Always include the first and last node.
        split_indices = [0]
        for i in range(1, len(node_ids) - 1):
            # If the node appears in more than one way, treat it as an intersection.
            if self.node_usage.get(node_ids[i], 0) > 1:
                split_indices.append(i)
        if split_indices[-1] != len(node_ids) - 1:
            split_indices.append(len(node_ids) - 1)

        # Split the way into segments between consecutive splitting nodes.
        for start_idx, end_idx in zip(split_indices, split_indices[1:]):
            if start_idx == end_idx:
                continue  # skip degenerate segments
            segment_node_ids = node_ids[start_idx:end_idx + 1]
            segment_coords = coords[start_idx:end_idx + 1]
            geom_segment = LineString(segment_coords)

            # The segment’s endpoints.
            source_osm = segment_node_ids[0]
            target_osm = segment_node_ids[-1]

            # Map OSM node ids to new node ids.
            for osm_node in [source_osm, target_osm]:
                if osm_node not in self.new_node_mapping:
                    new_id = len(self.new_node_mapping) + 1
                    self.new_node_mapping[osm_node] = new_id
                    self.node_features.append({
                        "id": new_id,
                        "osm_id": osm_node,
                        "geometry": Point(self.nodes[osm_node])
                    })
            source_new = self.new_node_mapping[source_osm]
            target_new = self.new_node_mapping[target_osm]

            # Create a new edge record.
            self.edges.append({
                "id": self.edge_id_counter,
                "osm_way_id": w.id,
                "source": source_new,
                "target": target_new,
                "highway": highway,
                "base_speed": base_speed,
                "priority": priority,
                "oneway": oneway,
                "geometry": geom_segment
            })
            self.edge_id_counter += 1

# -------------------------------------------------------------------------
# Main function.
def main(osm_pbf_file, output_gpkg):
    print("Counting node usage (to identify intersections)...")
    count_handler = CountHandler()
    # No need for locations in this pass.
    count_handler.apply_file(osm_pbf_file, locations=False)
    node_usage = count_handler.node_usage
    print(f"Counted node usage for {len(node_usage)} nodes.")

    print("Processing roads and splitting into segments...")
    segment_handler = RoadSegmentHandler(node_usage)
    # This pass requires node locations.
    segment_handler.apply_file(osm_pbf_file, locations=True)
    print(f"Created {len(segment_handler.edges)} edges from the road network.")

    # Create GeoDataFrames for edges and nodes.
    gdf_edges = gpd.GeoDataFrame(segment_handler.edges)
    gdf_nodes = gpd.GeoDataFrame(segment_handler.node_features)

    # Set the CRS to WGS84 and then reproject to EPSG:28992 (a meter-based CRS).
    gdf_edges.set_crs(epsg=4326, inplace=True)
    gdf_nodes.set_crs(epsg=4326, inplace=True)
    gdf_edges = gdf_edges.to_crs(epsg=28992)
    gdf_nodes = gdf_nodes.to_crs(epsg=28992)

    # Compute segment length (in meters) and travel cost (in seconds).
    # Note: base_speed is in km/h; convert to m/s by dividing by 3.6.
    gdf_edges["length_m"] = gdf_edges.geometry.length
    gdf_edges["cost"] = gdf_edges["length_m"] * 3.6 / gdf_edges["base_speed"] * gdf_edges["priority"]

    # ---------------------------------------------------------------
    # Remove disconnected areas (components) with fewer than 50 edges.
    # Build an undirected graph from the edge source/target nodes.
    G = nx.Graph()
    for _, row in gdf_edges.iterrows():
        G.add_edge(row["source"], row["target"])
    
    # Identify valid nodes in connected components with at least 50 edges.
    valid_nodes = set()
    for comp in nx.connected_components(G):
        # Select edges whose both endpoints belong to the component.
        comp_edges = gdf_edges[
            gdf_edges["source"].isin(comp) & gdf_edges["target"].isin(comp)
        ]
        if len(comp_edges) >= 50:
            valid_nodes.update(comp)
    
    # Filter edges and nodes to only include valid nodes.
    gdf_edges = gdf_edges[
        gdf_edges["source"].isin(valid_nodes) & gdf_edges["target"].isin(valid_nodes)
    ]
    gdf_nodes = gdf_nodes[gdf_nodes["id"].isin(valid_nodes)]
    # ---------------------------------------------------------------

    print("Writing output to GeoPackage:", output_gpkg)
    # Write the edges layer.
    gdf_edges.to_file(output_gpkg, layer="edges", driver="GPKG")
    # Write the nodes layer.
    gdf_nodes.to_file(output_gpkg, layer="nodes", driver="GPKG")
    print("Finished writing output.")

# -------------------------------------------------------------------------
# Command-line interface.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse an OSM PBF file and export a road network graph (with split segments, node connectors, segment lengths, travel cost, and removal of small disconnected areas) to a GeoPackage."
    )
    parser.add_argument("input", help="Input OSM PBF file (e.g. data.osm.pbf)")
    parser.add_argument("output", help="Output GeoPackage file (e.g. network.gpkg)")
    args = parser.parse_args()
    main(args.input, args.output)
