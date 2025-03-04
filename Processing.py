import ezdxf
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

# Load points from a DXF file
def extract_points_from_dxf(file_path):
    """Reads a DXF file and extracts points."""
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    points = []

    for entity in msp:
        print(entity.dxftype())  # Display the entity type

    for entity in msp.query('POINT'):
        point = (entity.dxf.location.x, entity.dxf.location.y)  # Use location
        points.append(point)

    return points

# Merge points that are too close
def merge_close_points(points, threshold=5):
    """Merges points that are too close into one by taking the average."""
    if not points:
        return []

    tree = KDTree(points)
    clusters = []
    visited = set()

    for i, point in enumerate(points):
        if i in visited:
            continue
        indices = tree.query_ball_point(point, threshold)
        if len(indices) > 1:  # Merge only if more than one point is found
            cluster = np.mean([points[j] for j in indices], axis=0)  # Average of cluster points
            clusters.append(tuple(cluster))
        else:
            clusters.append(point)  # Keep the original point if it is alone
        visited.update(indices)

    return clusters

# Sort points to form a logical contour
def order_points(points):
    """Sorts points to form a logical contour."""
    if not points:
        return []

    points = points[:]  # Copy points to avoid modifying the original
    ordered = [points.pop(0)]  # Start with the first point
    while points:
        last_point = ordered[-1]
        distances = cdist([last_point], points)  # Calculate distances to other points
        nearest_index = np.argmin(distances)  # Find the closest point
        ordered.append(points.pop(nearest_index))

    return ordered

# Save the closed polygon to a DXF file with a top-to-bottom mirror effect
def save_polygon_to_dxf(points, output_path):
    """Creates a DXF file with a closed polygon from processed points with a top-to-bottom mirror effect."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    if points:
        points.append(points[0])  # Close the polygon
        msp.add_lwpolyline(points, close=True)  # Add a closed polygon

    doc.saveas(output_path)

def calculate_polygon_area(points, unit='unit'):
    """Calculates the area of a polygon from ordered points and displays the unit."""
    if len(points) < 3:  # A polygon must have at least 3 points
        return 0

    area = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]  # The next point, looping back to the start
        area += x1 * y2 - y1 * x2

    area = abs(area) / 2
    area_cm2 = area / 100  # Convert area from mm^2 to cm^2
    print(f"🔲 {area_cm2:.1f} polygon area in cm^2")
    return area

# Use in the complete pipeline
def process_dxf(input_dxf, output_dxf, threshold=5, unit='unit'):
    """Executes the complete DXF file processing pipeline."""
    print("Reading DXF file...")
    points = extract_points_from_dxf(input_dxf)

    print(f"{len(points)} points extracted.")
    
    print("Merging points that are too close...")
    filtered_points = merge_close_points(points, threshold)
    print(f"{len(filtered_points)} points after merging.")

    print("Sorting points to form a contour...")
    ordered_points = order_points(filtered_points)

    calculate_polygon_area(ordered_points, unit)

    print("Saving the polygon to a new DXF file...")
    save_polygon_to_dxf(ordered_points, output_dxf)
    
    print(f"Polygon saved in {output_dxf}")

# Execution
input_dxf = "output.dxf"  # Replace with your input file
output_dxf = "output2.dxf"  # Name of the output file
unit = 'mm'  # Unit of the coordinates of the points in the DXF file

process_dxf(input_dxf, output_dxf, threshold=5, unit=unit)