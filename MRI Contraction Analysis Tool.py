import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import splprep, splev
import json
import cv2
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.path import Path
from scipy.optimize import curve_fit
import hashlib
from scipy import interpolate

# Constants
N_REGIONS = 20  # number of regions for contraction analysis
OUTPUT_DPI = 600  # resolution for saved figures

# analyse contraction
def export_wave_data(wave_interpolated_points, region_boundaries, 
                    point_spacing_mm, frame_duration_s, curvature_type="lesser"):
    """
    check if a wave stictly passes through a single region 
    """
    os.makedirs('results', exist_ok=True)
    
    region_speed_data = {}
    for i in range(len(region_boundaries) - 1):
        region_start = region_boundaries[i]
        region_end = region_boundaries[i + 1]
        # skip the initial and end region 
        if i == 0 or i == len(region_boundaries) - 2:
            continue
        region_speed_data[(region_start, region_end)] = []

    # process boundary point
    for wave_id, wave_data in wave_interpolated_points.items():
        boundary_points = wave_data['boundary_points']
        if len(boundary_points) < 2:
            continue  # at least two points can be used to calculate speed

        sorted_boundary_points = sorted(boundary_points, key=lambda x: x[1])
        
        # check if a wave passes through a single region
        for i in range(len(region_boundaries) - 1):
            region_start = region_boundaries[i]
            region_end = region_boundaries[i + 1]
            
            if i == 0 or i == len(region_boundaries) - 2:
                continue
                
            entry_point = None
            exit_point = None
            
            for t, p in sorted_boundary_points:
                if p >= region_start and p < region_end and entry_point is None:
                    entry_point = (t, p)
                elif p >= region_end and exit_point is None:
                    exit_point = (t, p)
                    break
            
            # 只有当有进入和离开点时才计算速度
            if entry_point and exit_point:
                t1, p1 = entry_point
                t2, p2 = exit_point
                
                # calculate velocity (mm/s)
                distance_mm = abs(p2 - p1) * point_spacing_mm
                time_s = abs(t2 - t1) * frame_duration_s
                speed = distance_mm / time_s if time_s > 0 else 0
                
                region_speed_data[(region_start, region_end)].append({
                    'wave_id': wave_id,
                    'speed': speed,
                    'time_start': t1 * frame_duration_s,
                    'time_end': t2 * frame_duration_s
                })

    # export CSV
    with open(f'results/{curvature_type}_region_speeds.csv', 'w') as f:
        f.write("region_id,region_start,region_end,avg_speed_mm_s,num_waves,wave_details\n")
        for region_id, ((start, end), speeds) in enumerate(region_speed_data.items(), 1):
            if speeds:
                avg_speed = np.mean([s['speed'] for s in speeds])
                wave_details = ";".join([
                    f"wave{s['wave_id']}:{s['speed']:.2f}:{s['time_start']:.2f}-{s['time_end']:.2f}"
                    for s in speeds
                ])
                num_waves = len(speeds)
            else:
                avg_speed = 0
                wave_details = ""
                num_waves = 0
            
            f.write(f"{region_id},{start},{end},{avg_speed:.2f},{num_waves},{wave_details}\n")

def calculate_wave_contraction_stats(contraction_sizes, wave_interpolated_points, region_boundaries):
    """
    calculate the contraction size of wave that through the region
    """
    # Initialize region data storage
    region_wave_data = {}
    
    # First identify which waves fully cross which regions
    valid_wave_regions = {}  # {wave_id: [valid_region_keys]}
    
    for wave_id, wave_data in wave_interpolated_points.items():
        boundary_points = wave_data['boundary_points']
        if len(boundary_points) < 2:
            continue
            
        # Sort boundary points by position
        sorted_boundary = sorted(boundary_points, key=lambda x: x[1])
        
        valid_regions = []
        for i in range(len(region_boundaries) - 1):
            region_start = region_boundaries[i]
            region_end = region_boundaries[i + 1]
            
            # Skip first and last regions
            if i == 0 or i == len(region_boundaries) - 2:
                continue
                
            # Check if wave enters and exits this region
            has_entry = False
            has_exit = False
            
            for t, p in sorted_boundary:
                if not has_entry and region_start <= p < region_end:
                    has_entry = True
                elif has_entry and p >= region_end:
                    has_exit = True
                    break
                    
            if has_entry and has_exit:
                valid_regions.append((region_start, region_end))
                
        valid_wave_regions[wave_id] = valid_regions
    
    # Now calculate contraction stats only for valid wave-region pairs
    for wave_id, wave_data in wave_interpolated_points.items():
        if wave_id not in valid_wave_regions:
            continue
            
        valid_regions = valid_wave_regions[wave_id]
        internal_points = wave_data['internal_points']
        
        for region_key in valid_regions:
            # Initialize this region if needed
            if region_key not in region_wave_data:
                region_wave_data[region_key] = {
                    'wave_contractions': {}
                }
            
            # Initialize this wave if needed
            if wave_id not in region_wave_data[region_key]['wave_contractions']:
                region_wave_data[region_key]['wave_contractions'][wave_id] = []
            
            # Add contraction values for points in this region
            if region_key in internal_points:
                for time, pos in internal_points[region_key]:
                    frame = int(round(time))
                    pos_rounded = int(round(pos))
                    
                    if (0 <= frame < contraction_sizes.shape[0] and 
                        0 <= pos_rounded < contraction_sizes.shape[1]):
                        region_wave_data[region_key]['wave_contractions'][wave_id].append(
                            contraction_sizes[frame, pos_rounded]
                        )
    
    return region_wave_data

def export_region_wave_contraction_stats(contraction_sizes, wave_interpolated_points, 
                                       region_boundaries, curvature_type="lesser"):
    """
    Export wave contraction statistics for each region using pre-calculated points.
    
    Args:
        contraction_sizes: Contraction size matrix
        wave_interpolated_points: Dictionary with pre-calculated points
        region_boundaries: Array of region boundary positions
        curvature_type: "lesser" or "greater"
    """
    os.makedirs('results', exist_ok=True)
    
    # Calculate wave contraction statistics using pre-calculated points
    region_wave_data = calculate_wave_contraction_stats(
        contraction_sizes=contraction_sizes, 
        wave_interpolated_points=wave_interpolated_points,
        region_boundaries=region_boundaries
    )
    
    # Prepare data for export
    region_stats = []
    for region_id, (region_key, data) in enumerate(sorted(region_wave_data.items()), 1):
        region_start, region_end = region_key
        wave_contractions = data['wave_contractions']
        num_waves = len(wave_contractions)
        
        # Calculate overall average contraction across all waves
        all_contractions = []
        for wave_id, contractions in wave_contractions.items():
            all_contractions.extend(contractions)
        
        if all_contractions:
            avg_contraction = np.mean(all_contractions)
        else:
            avg_contraction = 0
        
        # Create wave details string with wave ID and its average contraction
        wave_details = []
        for wave_id, contractions in sorted(wave_contractions.items()):
            if contractions:
                wave_avg = np.mean(contractions)
                wave_details.append(f"wave{wave_id}:{wave_avg:.4f}")
            else:
                wave_details.append(f"wave{wave_id}:0.0000")
        
        region_stats.append({
            'region_id': region_id,
            'region_start': region_start,
            'region_end': region_end,
            'avg_wave_contraction_mm': avg_contraction,
            'num_waves': num_waves,
            'wave_details': ';'.join(wave_details)
        })
    
    # Export to CSV
    filename = f'results/{curvature_type}_region_wave_contraction_size.csv'
    with open(filename, 'w') as f:
        f.write("region_id,region_start,region_end,avg_wave_contraction_mm,num_waves,wave_details\n")
        
        for stat in region_stats:
            f.write(f"{stat['region_id']},{stat['region_start']},{stat['region_end']},")
            f.write(f"{stat['avg_wave_contraction_mm']:.4f},{stat['num_waves']},")
            f.write(f"{stat['wave_details']}\n")
    
    print(f"Region wave contraction statistics exported to {filename}")

class MRISegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MRI Contraction Analysis Tool")
        self.root.geometry("1000x800")

        self.workspace_path = None
        # Variables
        self.image_folder = None
        self.image_files = []
        self.current_index = 0
        self.images = []

        # ROI related variables
        self.roi_points = {}  # Dictionary to store points for each frame
        self.control_points = []  # Current control points (always a list)
        self.spline_points = []  # Current spline points
        self.spline_line = None  # Current spline line
        self.control_point_markers = []  # Current control point markers
        self.selected_point = None  # Currently selected control point
        self.num_spline_points = 1000  # Number of points in the spline
        self.is_drawing = False  # Flag to indicate if we're in drawing mode
        self.is_editing = False  # Flag to indicate if we're in editing mode

        self.spline_points_per_frame = {}  # {frame_idx: array of 398 points per frame}
        self.show_split_points = True      # Show split points by default
        self.show_control_points = True    # Show control points by default
        self.show_midpoints = True         # Show midpoints by default
        self.centerline_plots = []         # Store the plotted centerlines
        self.show_centerline = True  # Show the centerline by default

        # key points (distal and proximal)
        self.key_points = {}  # {frame_idx: {'major': {'index': int, 'position': [x,y]}, 'minor': ...}}
        self.current_key_type = None  
        self.keypoint_colors = {'major': 'gold', 'minor': 'cyan'}  # colour
        self.keypoint_size = 10  # the size of key points

        # zoom
        self.zoom_scale = 1.0  
        self.default_zoom = 1.0  
        self.zoom_step = 0.1  # Zoom step per mouse wheel tick
        self.xlim = None  
        self.ylim = None  

        # Undo stack
        self.undo_stack = []  # Stack to store previous states for undo

        # Auto propagation variables
        self.auto_propagating = False  # Flag to indicate if we're in auto propagation mode
        self.propagation_step = 1  # Number of frames to propagate forward

        # Brightness adjustment
        self.brightness_scale = 1.0  # Default brightness scale

        # Create UI elements
        self.create_menu()
        self.create_widgets()
        self.setup_interactions()

        # Optical flow related variables
        self.flow_quiver = []  
        self.flow_colorbar = None  
        self.show_flow = False  

        # colorbar
        self.cax = None
        
        self.standard_position = [0.125, 0.11, 0.775, 0.77]  # position of matplotlib

        # analysis parameters of contraction
        self.lesser_spacing = tk.DoubleVar(value=1.0)  # mm between points on lesser curvature
        self.greater_spacing = tk.DoubleVar(value=1.0)  # mm between points on greater curvature
        self.frame_duration = tk.DoubleVar(value=0.475)  # seconds per frame
        self.max_contraction = tk.DoubleVar(value=10.0)  # maximum contraction size for visualization

        # Pixel spacing 
        self.pixel_spacing = (1.0, 1.0)  # default value (spacing_x, spacing_y) in mm/pixel
        self.has_pixel_spacing = False

    def save_workspace(self):
        """save data of workspace"""
        if not self.image_files:
            messagebox.showwarning("warning", "No workspace available to save")
            return

        try:
            # calculate hashes of file 
            file_hashes = {}
            for img_file in self.image_files:
                file_hash = self._get_file_hash(img_file)
                if file_hash is None:
                    raise ValueError(f"can't calculate hashes: {img_file}")
                file_hashes[os.path.basename(img_file)] = file_hash
            print("Saving workspace, checking the data...")
            
            # Enhanced safe conversion function
            def convert_for_json(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (list, tuple)):
                    return [convert_for_json(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                return obj

            # construct the structure of data
            workspace_data = {
                "image_folder": os.path.relpath(self.image_folder) if self.image_folder else None,
                "file_hashes": file_hashes,
                "current_index": self.current_index,
                "roi_points": {str(k): convert_for_json(v) for k, v in self.roi_points.items()},
                "control_points": convert_for_json(self.control_points),
                "spline_points": convert_for_json(self.spline_points),
                "is_drawing": self.is_drawing,
                "is_editing": self.is_editing,
                "num_spline_points": self.num_spline_points,
                "spline_points_per_frame": {
                    str(k): {
                        'points': convert_for_json(v['points']),
                        'point_types': convert_for_json(v['point_types']),
                        'major_keypoint': convert_for_json(v['major_keypoint']),
                        'minor_keypoint': convert_for_json(v['minor_keypoint']),
                        'midpoints': self._serialize_midpoints(v.get('midpoints'))
                    }
                    for k, v in self.spline_points_per_frame.items()
                },
                "show_centerline": self.show_centerline, 

                "contraction_parameters": {
                    "lesser_spacing": self.lesser_spacing.get(),
                    "greater_spacing": self.greater_spacing.get(),
                    "frame_duration": self.frame_duration.get(),
                    "max_contraction": self.max_contraction.get(),
                    "pixel_spacing": list(self.pixel_spacing),
                    "has_pixel_spacing": self.has_pixel_spacing
                }
            }
            
            if hasattr(self, 'key_points') and self.key_points:
                workspace_data["key_points"] = convert_for_json(self.key_points)

            if hasattr(self, 'centerline_data'):
                workspace_data["centerline_data"] = {
                    'fitted_coefficients': convert_for_json(self.centerline_data['fitted_coefficients']),
                    'covariance_matrix': convert_for_json(self.centerline_data['covariance_matrix']),
                    'centerline_points': convert_for_json(self.centerline_data['centerline_points']),
                    'frame_indices': convert_for_json(self.centerline_data['frame_indices']),
                    'original_midpoints': [convert_for_json(arr) for arr in self.centerline_data['original_midpoints']]
                }

            print("Data validation passed, starting to write to file...")
            
            # write
            save_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON file", "*.json"), ("all files", "*.*")],
                title="save workspace"
            )
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(workspace_data, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Saved successfully", f"Workspace saved to:\n{save_path}")
                self.status_var.set(f"Workspace saved - {os.path.basename(save_path)}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Save failed",
                f"error: {str(e)}\n\n"
                "Please check"
            )

    def _serialize_midpoints(self, midpoints):
        """Dedicated serialization for midpoint data"""
        if midpoints is None:
            return None
        
        result = {}
        for category in ['all_midpoints', 'valid_midpoints']:
            if category in midpoints and midpoints[category]:
                result[category] = {
                    'points': [list(map(float, point)) for point in midpoints[category]['points']],
                    'is_keypoint': [bool(x) for x in midpoints[category].get('is_keypoint', [])],
                    'endpoints': [
                        [list(map(float, ep1)), list(map(float, ep2))]
                        for ep1, ep2 in midpoints[category].get('endpoints', [])
                    ]
                }
            else:
                result[category] = None
        return result
    
    def load_workspace(self):
        try:
            load_path = filedialog.askopenfilename(
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                title="Load Workspace"
            )
            if not load_path:
                return
            
            self.workspace_path = load_path

            with open(load_path, 'r') as f:
                workspace_data = json.load(f)
            
            # try the original path
            image_folder = workspace_data["image_folder"]
            if image_folder and not os.path.isabs(image_folder):
                workspace_dir = os.path.dirname(load_path)
                image_folder = os.path.normpath(os.path.join(workspace_dir, image_folder))
            
            # user selection
            if not os.path.exists(image_folder):
                messagebox.showwarning("Path does not exist", "Could not locate the image folder automatically. Please select manually.")
                image_folder = filedialog.askdirectory(title="Select Image Folder")
                if not image_folder:
                    return False
            
            # verify the images
            verified_files = self._verify_files(image_folder, workspace_data["file_hashes"])
            if len(verified_files) != len(workspace_data["file_hashes"]):
                messagebox.showwarning("Warning", f"Found {len(verified_files)}/{len(workspace_data['file_hashes'])} images \n Please check the files")
            
            self.image_folder = image_folder
            self.image_files = verified_files

            # 1. load basic data
            #self.image_folder = workspace_data.get("image_folder")
            self.current_index = int(workspace_data.get("current_index", 0))
            
            # 2. load ROI data 
            self.roi_points = {}
            if "roi_points" in workspace_data:
                for idx, points in workspace_data["roi_points"].items():
                    self.roi_points[int(idx)] = np.array(points) if points else np.array([])

            # 3. load control and spline point
            self.control_points = list(workspace_data.get("control_points", []))
            spline_data = workspace_data.get("spline_points", [])
            self.spline_points = np.array(spline_data) if spline_data else []

            # 4. load key points
            self.key_points = {}
            if "key_points" in workspace_data:
                for frame_idx, kp_data in workspace_data["key_points"].items():
                    frame_idx = int(frame_idx)
                    self.key_points[frame_idx] = {}
                    for kp_type, kp_info in kp_data.items():
                        self.key_points[frame_idx][kp_type] = {
                            'index': kp_info['index'],
                            'position': np.array(kp_info['position'])
                        }

            # 5. load 
            if "spline_points_per_frame" in workspace_data:
                self.spline_points_per_frame = {}
                for k, v in workspace_data["spline_points_per_frame"].items():
                    frame_idx = int(k)
                    self.spline_points_per_frame[frame_idx] = {
                        'points': np.array(v['points']),
                        'point_types': np.array(v['point_types']),
                        'major_keypoint': np.array(v['major_keypoint']),
                        'minor_keypoint': np.array(v['minor_keypoint']),
                        'midpoints': self._deserialize_midpoints(v.get('midpoints'))
                    }

            # 6. load centerline data
            if "centerline_data" in workspace_data:
                centerline_data = workspace_data["centerline_data"]
                self.centerline_data = {
                    'fitted_coefficients': np.array(centerline_data['fitted_coefficients']),
                    'covariance_matrix': np.array(centerline_data['covariance_matrix']),
                    'centerline_points': np.array(centerline_data['centerline_points']),
                    'frame_indices': np.array(centerline_data['frame_indices']),
                    'original_midpoints': [np.array(arr) for arr in centerline_data['original_midpoints']]
                }

            # 6. load status
            self.is_drawing = workspace_data.get("is_drawing", False)
            self.is_editing = workspace_data.get("is_editing", False)
            self.num_spline_points = workspace_data.get("num_spline_points", 1000)

            # 7. 
            self.spline_line = None
            self.control_point_markers = []
            self.selected_point = None
            self.current_key_type = None
            self.centerline_plots = []  

            # 8. load images
            self.load_images()
            
            if self.current_index in self.roi_points:
                self.control_points = self.roi_points[self.current_index].tolist()

            # 9. update interface
            self.display_current_image()
            self.update_counter()
            self.update_button_states()

            # 10. 设置状态提示
            if self.is_drawing:
                self.mode_var.set("Drawing ROI")
                self.status_var.set("Drawing ROI (right-click mouse)")
            elif self.is_editing:
                self.mode_var.set("Editing ROI")
                self.status_var.set("Editing ROI: Drag control points or right-click to add/delete.")
            else:
                self.mode_var.set("View")
                loaded_kp = len(self.key_points.get(self.current_index, {}))
                self.status_var.set(
                    f"Loaded workspace | number of key points at curent frame: {loaded_kp}/2" + 
                    (" (需补全)" if loaded_kp < 2 else "")
                )

            # load contraction parameters
            if "contraction_parameters" in workspace_data:
                params = workspace_data["contraction_parameters"]
                self.lesser_spacing.set(float(params.get("lesser_spacing", 1.17)))
                self.greater_spacing.set(float(params.get("greater_spacing", 1.56)))
                self.frame_duration.set(float(params.get("frame_duration", 0.475)))
                self.max_contraction.set(float(params.get("max_contraction", 10.0)))
                pixel_spacing = params.get("pixel_spacing", [1.0, 1.0])
                self.pixel_spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]))
                self.has_pixel_spacing = params.get("has_pixel_spacing", False)

            messagebox.showinfo(
                "Load Successful",
                f"Workspace loaded:\n"
                f"- Images: {len(self.image_files)} frames\n"
                f"- ROI: {len(self.roi_points)} frames\n"
                f"- Key points: {sum(len(v) for v in self.key_points.values())} points\n"
                f"- Centerline: {'Loaded' if hasattr(self, 'centerline_data') else 'None'}"
            )


        except json.JSONDecodeError:
            messagebox.showerror("Error", "File format error: Not a valid JSON file")
        except KeyError as e:
            messagebox.showerror("Error", f"Missing required field: {str(e)}")
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to load:\n{str(e)}\n\n"
                "Possible reasons:\n"
                "1. File was modified or corrupted\n"
                "2. Version incompatibility\n"
                "3. File path contains special characters"
            )
            
            self.clear_roi()
            self.key_points.clear()

    def _deserialize_midpoints(self, midpoints_data):
        if not midpoints_data:
            return None
        
        deserialized = {}
        for category in ['all_midpoints', 'valid_midpoints']:
            if category in midpoints_data and midpoints_data[category]:
                cat_data = midpoints_data[category]
                deserialized[category] = {
                    'points': np.array(cat_data['points']),
                    'is_keypoint': np.array(cat_data['is_keypoint']),
                    'endpoints': np.array(
                        [(ep1, ep2) for ep1, ep2 in cat_data['endpoints']],
                        dtype=[('endpoint1', 'f8', 2), ('endpoint2', 'f8', 2)]
                    )
                }
            else:
                deserialized[category] = None
        
        if 'contour_points' in midpoints_data and midpoints_data['contour_points']:
            deserialized['contour_points'] = np.array(midpoints_data['contour_points'])
        
        return deserialized

    def _get_file_hash(self, filepath):
        """Compute file hash"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"Compute file hash failed: {str(e)}")
            return None

    def _find_image_folder_by_hashes(self, file_hashes, folder_name=None):
        """Find image folder by hash value"""
        search_locations = [
            os.path.dirname(self.workspace_path),  # Directory of the workspace file
            os.path.expanduser("~"),               # user main directory 
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Desktop"),
        ]
        
        if folder_name:
            for loc in search_locations:
                possible_path = os.path.join(loc, folder_name)
                if os.path.exists(possible_path):
                    if self._verify_files(possible_path, file_hashes):
                        return possible_path
        
        # further search
        for loc in search_locations:
            for root, dirs, files in os.walk(loc):
                matched = 0
                for file in files:
                    if file in file_hashes:
                        filepath = os.path.join(root, file)
                        if self._get_file_hash(filepath) == file_hashes[file]:
                            matched += 1
                            if matched == len(file_hashes):
                                return root
        return None

    def _verify_files(self, folder, file_hashes):
        """Verify if files in the folder match the hash values"""
        verified_files = []
        for filename, expected_hash in file_hashes.items():
            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                if self._get_file_hash(filepath) == expected_hash:
                    verified_files.append(filepath)
        return sorted(verified_files)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Folder", command=self.load_image_folder)
        #file_menu.add_command(label="Save ROIs", command=self.save_rois)
        file_menu.add_command(label="Save Workspace", command=self.save_workspace)
        file_menu.add_command(label="Load Workspace", command=self.load_workspace)
        file_menu.add_command(label="Export Images with ROIs", command=self.export_images_with_rois)  
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        roi_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ROI", menu=roi_menu)
        roi_menu.add_command(label="Start Drawing ROI", command=self.start_drawing_roi)
        roi_menu.add_command(label="Complete ROI", command=self.complete_roi)
        roi_menu.add_command(label="Clear ROI", command=self.clear_roi)
        roi_menu.add_command(label="Edit ROI", command=self.toggle_edit_mode)
        roi_menu.add_command(label="Copy ROI to Next Frame", command=self.copy_roi_to_next)

        roi_menu.add_separator()
        roi_menu.add_command(label="Mark Major Bend", command=lambda: self.set_keypoint_type('major'))
        roi_menu.add_command(label="Mark Minor Bend", command=lambda: self.set_keypoint_type('minor'))
        roi_menu.add_command(label="Clear Key Points", command=self.clear_key_points)

        # Add new menu options for propagation
        roi_menu.add_separator()
        roi_menu.add_command(label="Propagate ROI with Optical Flow", command=self.propagate_roi_with_flow)
        roi_menu.add_command(label="Auto Propagate to All Frames", command=self.auto_propagate_all_frames)

        roi_menu.add_separator()
        roi_menu.add_command(label="Save Wall Points", command=self.save_splinerois)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Centerline", menu=analysis_menu)
        analysis_menu.add_command(label="Fit Centerline", command=self.fit_polynomial_to_midpoints)
        analysis_menu.add_checkbutton(label="Toggle Centerline", command=self.toggle_centerline,variable=tk.BooleanVar(value=self.show_centerline))
        analysis_menu.add_command(label="Save Centerline Data", command=self.save_centerline_data)

        contraction_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Contraction Analysis", menu=contraction_menu)
        contraction_menu.add_command(label="Lesser Contraction Map", command=lambda: self.analyze_contraction("lesser"))
        contraction_menu.add_command(label="Greater Contraction Map", command=lambda: self.analyze_contraction("greater"))
        contraction_menu.add_command(label="Manual Wave Tracking", command=self.manual_wave_tracking)  
        contraction_menu.add_command(label="Contraction Analysis Parameters", command=self.show_parameter_dialog)

    def set_pixel_spacing(self, spacing_x=None, spacing_y=None):
        if spacing_x is None or spacing_y is None:
            from tkinter import simpledialog
            
            spacing_x = simpledialog.askfloat("Pixel Spacing", 
                                            "Enter pixel spacing in X direction (mm/pixel):", initialvalue=1.0, parent=self.root)
            spacing_y = simpledialog.askfloat("Pixel Spacing", 
                                            "Enter pixel spacing in Y direction (mm/pixel):", initialvalue=1.0, parent=self.root)
            
            if spacing_x is None or spacing_y is None:
                messagebox.showwarning("Warning", "Using default pixel spacing (1.0, 1.0)")
                spacing_x, spacing_y = 1.0, 1.0
        
        self.pixel_spacing = (float(spacing_x), float(spacing_y))
        self.has_pixel_spacing = (spacing_x != 1.0 or spacing_y != 1.0)
        
        messagebox.showinfo("Pixel Spacing Set", f"Pixel spacing set to: X={spacing_x:.4f} mm, Y={spacing_y:.4f} mm")
    
    def show_parameter_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Contraction Analysis Parameters")
        dialog.geometry("450x450")  
        dialog.transient(self.root)
        dialog.grab_set()

        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() - width) // 2
        y = (dialog.winfo_screenheight() - height) // 2
        dialog.geometry(f"+{x}+{y}")
        
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Pixel spacing 
        pixel_spacing_frame = ttk.LabelFrame(scrollable_frame, text="Pixel Spacing Settings (mm/pixel)")
        pixel_spacing_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(pixel_spacing_frame, text="X Spacing:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        x_entry = ttk.Entry(pixel_spacing_frame, width=12)
        x_entry.insert(0, str(self.pixel_spacing[0] if self.pixel_spacing else 1.0))
        x_entry.grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(pixel_spacing_frame, text="Y Spacing:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        y_entry = ttk.Entry(pixel_spacing_frame, width=12)
        y_entry.insert(0, str(self.pixel_spacing[1] if self.pixel_spacing else 1.0))
        y_entry.grid(row=1, column=1, pady=5, padx=5)

        def update_pixel_spacing():
            try:
                x_val = float(x_entry.get())
                y_val = float(y_entry.get())
                if x_val <= 0 or y_val <= 0:
                    raise ValueError("Pixel spacing values must be positive")
                self.pixel_spacing = (x_val, y_val)
                self.has_pixel_spacing = (x_val != 1.0 or y_val != 1.0)
                messagebox.showinfo("Success", f"Pixel spacing updated to: X={x_val:.4f} mm, Y={y_val:.4f} mm")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {str(e)}")
        
        update_button = ttk.Button(pixel_spacing_frame, text="Update", command=update_pixel_spacing)
        update_button.grid(row=2, column=0, columnspan=2, pady=5)

        avergae_length_frame = ttk.LabelFrame(scrollable_frame, text="Average length (mm/point)")
        avergae_length_frame.pack(fill=tk.X, pady=5, padx=5)
        
        calc_button_frame = ttk.Frame(avergae_length_frame)
        calc_button_frame.grid(row=0, column=0, columnspan=2, pady=10)
        ttk.Button(calc_button_frame, text="Calculate Average Length", command=self.calculate_average_lengths).pack(side=tk.LEFT, padx=5)

        ttk.Label(avergae_length_frame, text="Lesser Curvature Point Spacing:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        lesser_entry = ttk.Entry(avergae_length_frame, width=12)
        lesser_entry.insert(0, str(self.lesser_spacing.get()))
        lesser_entry.grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(avergae_length_frame, text="Greater Curvature Point Spacing:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        greater_entry = ttk.Entry(avergae_length_frame, width=12)
        greater_entry.insert(0, str(self.greater_spacing.get()))
        greater_entry.grid(row=2, column=1, pady=5, padx=5)

        analysis_frame = ttk.LabelFrame(scrollable_frame, text="Other Parameters")
        analysis_frame.pack(fill=tk.X, pady=10, padx=5)
        
        ttk.Label(analysis_frame, text="Frame Duration (seconds):").grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        duration_entry = ttk.Entry(analysis_frame)
        duration_entry.insert(0, str(self.frame_duration.get()))
        duration_entry.grid(row=2, column=1, pady=5, padx=5)
        
        ttk.Label(analysis_frame, text="Max Contraction Size (mm):").grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        max_entry = ttk.Entry(analysis_frame)
        max_entry.insert(0, str(self.max_contraction.get()))
        max_entry.grid(row=3, column=1, pady=5, padx=5)
        
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(pady=15)
        
        def save_parameters():
            try:
                x_val = float(x_entry.get())
                y_val = float(y_entry.get())
                if x_val <= 0 or y_val <= 0:
                    raise ValueError("Pixel spacing values must be positive")
                self.pixel_spacing = (x_val, y_val)
                self.has_pixel_spacing = (x_val != 1.0 or y_val != 1.0)

                lesser = float(lesser_entry.get())
                greater = float(greater_entry.get())
                duration = float(duration_entry.get())
                max_cont = float(max_entry.get())
                
                if lesser <= 0 or greater <= 0 or duration <= 0 or max_cont <= 0:
                    raise ValueError("All values must be positive")
                    
                self.lesser_spacing.set(lesser)
                self.greater_spacing.set(greater)
                self.frame_duration.set(duration)
                self.max_contraction.set(max_cont)
                
                dialog.destroy()
                messagebox.showinfo("Success", "Parameters saved successfully!")
                
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {str(e)}")
        
        def reset_to_default():
            # reset pixel spacing
            x_entry.delete(0, tk.END)
            x_entry.insert(0, "1.0")
            y_entry.delete(0, tk.END)
            y_entry.insert(0, "1.0")
            
            # reset other
            lesser_entry.delete(0, tk.END)
            lesser_entry.insert(0, "1.17")
            greater_entry.delete(0, tk.END)
            greater_entry.insert(0, "1.56")
            duration_entry.delete(0, tk.END)
            duration_entry.insert(0, "0.475")
            max_entry.delete(0, tk.END)
            max_entry.insert(0, "10.0")
        
        def cancel():
            dialog.destroy()

        #ttk.Button(button_frame, text="Calculate Average Length", command=self.calculate_average_lengths).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save", command=save_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Default", command=reset_to_default).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)
        
        dialog.wait_window()

    def analyze_contraction(self, curvature_type):
        try:
            self.show_contraction_parameters_check(curvature_type)
            
            centerline_file = filedialog.askopenfilename(
                title=f"Select Centerline File",
                filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
            )
            if not centerline_file:
                return
            
            wallpoints_file = filedialog.askopenfilename(
                title=f"Select Wall Points File",
                filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
            )
            if not wallpoints_file:
                return
            
            if curvature_type == "lesser":
                spacing = self.lesser_spacing.get()
            else:
                spacing = self.greater_spacing.get()
                
            ref_radii, cont_sizes, occlusion, wall_points, point_types = self.calculate_contraction(
                centerline_file, wallpoints_file, curvature_type, self.pixel_spacing)
            
            self.visualize_contraction_results(ref_radii, cont_sizes, wall_points, point_types, curvature_type)
            
            self.contraction_data = {
                'reference_radii': ref_radii,
                'contraction_sizes': cont_sizes,
                'percentage_occlusion': occlusion,
                'wall_points': wall_points,
                'point_types': point_types,
                'curvature_type': curvature_type
            }
            
            messagebox.showinfo("Success", f"{curvature_type.capitalize()} curvature analysis completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Contraction analysis failed: {str(e)}")

    def calculate_contraction(self, centerline_file, wallpoints_file, curvature_type="lesser", pixel_spacing=None):
        """
        Calculate gastric wall contraction distances.
        
        Args:
            centerline_file: Path to .npz file with single frame centerline data
            wallpoints_file: Path to .npz file with multiple frames of wall points
            curvature_type: "lesser" or "greater" curvature
            pixel_spacing: Tuple (spacing_x, spacing_y) in mm/pixel, or None for pixel distance
        
        Returns:
            reference_radii: Array of reference radii
            contraction_sizes: Contraction size matrix (time × point)
            wall_points: Filtered wall points
            point_types: Filtered point types
        """
        # Load centerline data
        centerline_data = np.load(centerline_file)
        centerline_points = centerline_data['centerline_points'] 
        
        # Reshape centerline if necessary
        if centerline_points.ndim == 3:
            if centerline_points.shape[0] == 1:
                centerline_points = centerline_points[0]  # Convert from (1, n, 2) to (n, 2)
            else:
                raise ValueError("Centerline data must contain only one frame")
        elif centerline_points.ndim != 2:
            raise ValueError("Invalid shape for centerline data")
        
        # Reverse centerline for greater curvature
        if curvature_type == "greater":
            centerline_points = centerline_points[::-1]
        
        n_center_points = centerline_points.shape[0]
        print(f"Number of centerline points: {n_center_points}")
        
        # Load wall point data
        wall_data = np.load(wallpoints_file)
        wall_points = wall_data['points']  # Shape: (n_frames, n_wall_points, 2)
        point_types = wall_data['point_types']  # Shape: (n_frames, n_wall_points)
        
        print(f"Wall point data shape: {wall_points.shape}")
        
        n_frames = wall_points.shape[0]
        
        # Define point types
        normal_point_type = 1 if curvature_type == "lesser" else 0
        key_point_type = 2 
        
        # Process wall points
        filtered_wall_points = []
        filtered_point_types = []
        
        for t in range(n_frames):
            mask = (point_types[t] == normal_point_type) | (point_types[t] == key_point_type)
            frame_points = wall_points[t][mask]
            frame_types = point_types[t][mask]
            
            key_points = frame_points[frame_types == key_point_type] 
            normal_points = frame_points[frame_types == normal_point_type]  
            
            if len(key_points) != 2:
                print(f"Warning: Frame {t} has {len(key_points)} key points (expected 2)")
                if len(key_points) < 2:
                    continue
            
            # Identify major and minor keypoints
            if key_points[0][0] > key_points[1][0]:
                major_keypoint = key_points[0]  
                minor_keypoint = key_points[1]  
            else:
                major_keypoint = key_points[1] 
                minor_keypoint = key_points[0] 
            
            if curvature_type == "lesser":
                combined_points = np.vstack([
                    [minor_keypoint],    # Start point (lesser curvature)
                    normal_points,       # Normal points
                    [major_keypoint]     # End point (greater curvature)
                ])
            else:  # greater curvature
                combined_points = np.vstack([
                    [major_keypoint],    # Start point (greater curvature)
                    normal_points,       # Normal points
                    [minor_keypoint]     # End point (lesser curvature)
                ])
            
            filtered_wall_points.append(combined_points)

            # Create point type array
            filtered_types = np.zeros(n_center_points, dtype=int)
            filtered_types[0] = 2  # Start keypoint
            filtered_types[-1] = 2  # End keypoint
            filtered_types[1:-1] = 1  # Normal points
            
            filtered_point_types.append(filtered_types)
        
        # Convert to numpy arrays
        filtered_wall_points = np.array(filtered_wall_points)
        filtered_point_types = np.array(filtered_point_types)
        
        # Calculate distances matrix
        distances = np.zeros((n_frames, n_center_points))
        for t in range(n_frames):
            for n in range(n_center_points):
                # Calculate pixel differences
                dx_pixels = filtered_wall_points[t, n][0] - centerline_points[n][0]
                dy_pixels = filtered_wall_points[t, n][1] - centerline_points[n][1]
                
                if pixel_spacing is not None:
                    spacing_x, spacing_y = pixel_spacing
                    # Convert to physical distance (consider anisotropic spacing)
                    dx_mm = dx_pixels * spacing_x
                    dy_mm = dy_pixels * spacing_y
                    distances[t, n] = np.sqrt(dx_mm**2 + dy_mm**2)  # Physical distance (mm)
                else:
                    # Use pixel distance
                    distances[t, n] = np.sqrt(dx_pixels**2 + dy_pixels**2)  # Pixel distance
        
        # Calculate reference radius and contraction sizes
        reference_radii = np.mean(distances, axis=0)
        contraction_sizes = np.maximum(reference_radii - distances, 0)

        percentage_occlusion = np.zeros_like(contraction_sizes)
        for n in range(n_center_points):
            percentage_occlusion[:, n] = (contraction_sizes[:, n] / reference_radii[n]) * 100
            if reference_radii[n] <= 0:  
                print("warning: please check the data")
        
        return reference_radii, contraction_sizes, percentage_occlusion, filtered_wall_points, filtered_point_types

    def show_contraction_parameters_check(self, curvature_type):
        if curvature_type == "lesser":
            spacing = self.lesser_spacing.get()
        else:
            spacing = self.greater_spacing.get()
        
        frame_duration = self.frame_duration.get()
        max_contraction = self.max_contraction.get()
        
        pixel_spacing_x = self.pixel_spacing[0] if self.pixel_spacing else 1.0
        pixel_spacing_y = self.pixel_spacing[1] if self.pixel_spacing else 1.0
        
        check_window = tk.Toplevel(self.root)
        check_window.title(f"{curvature_type.capitalize()} Curvature Parameters Check")
        check_window.geometry("600x700")
        check_window.transient(self.root)
        check_window.grab_set()

        check_window.update_idletasks()
        width = check_window.winfo_width()
        height = check_window.winfo_height()
        x = (check_window.winfo_screenwidth() - width) // 2
        y = (check_window.winfo_screenheight() - height) // 2
        check_window.geometry(f"+{x}+{y}")
        
        content_frame = tk.Frame(check_window, padx=30, pady=30)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(
            content_frame, 
            text=f"Please check {curvature_type.capitalize()} Curvature Parameters",
            font=("Arial", 16, "bold"),
            fg="blue"
        ) 
        title_label.pack(pady=(0, 30))

        params_frame = tk.Frame(content_frame)
        params_frame.pack(fill=tk.X, pady=20)

        spacing_label = tk.Label(params_frame, text=f"{spacing:.2f} mm", font=("Arial", 10))
        duration_label = tk.Label(params_frame, text=f"{frame_duration:.3f} s", font=("Arial", 10))
        contraction_label = tk.Label(params_frame, text=f"{max_contraction:.1f} mm", font=("Arial", 10))
        pixel_x_label = tk.Label(params_frame, text=f"{pixel_spacing_x:.4f} mm", font=("Arial", 10))
        pixel_y_label = tk.Label(params_frame, text=f"{pixel_spacing_y:.4f} mm", font=("Arial", 10))
        
        tk.Label(params_frame, text="Pixel Spacing X:", 
                font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=3, padx=5)  
        pixel_x_label.grid(row=0, column=1, sticky=tk.W, pady=3, padx=5)
        
        tk.Label(params_frame, text="Pixel Spacing Y:", 
                font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, pady=3, padx=5)
        pixel_y_label.grid(row=1, column=1, sticky=tk.W, pady=3, padx=5)
        
        tk.Label(params_frame, text="Point Spacing:", 
                font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=3, padx=5)
        spacing_label.grid(row=2, column=1, sticky=tk.W, pady=3, padx=5)
        
        tk.Label(params_frame, text="Frame Duration:", 
                font=("Arial", 10, "bold")).grid(row=3, column=0, sticky=tk.W, pady=3, padx=5)
        duration_label.grid(row=3, column=1, sticky=tk.W, pady=3, padx=5)
        
        tk.Label(params_frame, text="Max Contraction:", 
                font=("Arial", 10, "bold")).grid(row=4, column=0, sticky=tk.W, pady=3, padx=5)
        contraction_label.grid(row=4, column=1, sticky=tk.W, pady=3, padx=5)
        
        separator = ttk.Separator(content_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=10)

        note_text = (
            "Please carefully verify these parameters before proceeding with the contraction analysis.\n\n"
            "• Pixel Spacing: Physical size of each pixel (mm/pixel)\n"
            "• Point Spacing: Distance between adjacent points along the curvature\n"
            "• Frame Duration: Time interval between consecutive frames\n"
            "• Max Contraction: Maximum contraction size for visualization scaling\n\n"
        )
        note_label = tk.Label(
            content_frame, 
            text=note_text,
            font=("Arial", 12),
            justify=tk.LEFT,
            wraplength=500,
            padx=10
        )
        note_label.pack(pady=15)
        
        button_frame = tk.Frame(content_frame)
        button_frame.pack(pady=30)
        
        def proceed():
            check_window.destroy()
        
        def adjust_parameters():
            check_window.destroy()  
            self.show_parameter_dialog()  
            
            if curvature_type == "lesser":
                new_spacing = self.lesser_spacing.get()
            else:
                new_spacing = self.greater_spacing.get()
            
            spacing_label.config(text=f"{new_spacing:.2f} mm")
            duration_label.config(text=f"{self.frame_duration.get():.3f} s")
            contraction_label.config(text=f"{self.max_contraction.get():.1f} mm")
            pixel_x_label.config(text=f"{self.pixel_spacing[0]:.4f} mm")
            pixel_y_label.config(text=f"{self.pixel_spacing[1]:.4f} mm")
            
            check_window.deiconify()
        
        def cancel():
            check_window.destroy()
        
        proceed_button = tk.Button(button_frame, text="Proceed", command=proceed, 
                                width=15, height=2, font=("Arial", 11))
        adjust_button = tk.Button(button_frame, text="Adjust Parameters", command=adjust_parameters, 
                                width=15, height=2, font=("Arial", 11))
        
        proceed_button.pack(side=tk.LEFT, padx=5)
        adjust_button.pack(side=tk.LEFT, padx=5)
        
        check_window.wait_window()

    def visualize_contraction_results(self, reference_radii, contraction_sizes, wall_points, point_types, curvature_type="lesser"):
        try:
            n_frames = contraction_sizes.shape[0]
            n_points = contraction_sizes.shape[1]
            
            # construct new window
            result_window = tk.Toplevel(self.root)
            result_window.title(f"{curvature_type.capitalize()} Curvature Contraction Results")
            
            result_window.transient(self.root)  
            result_window.grab_set()  
            result_window.lift()  
            result_window.focus_force()  
            
            result_window.update_idletasks()
            width = result_window.winfo_width()
            height = result_window.winfo_height()
            x = (result_window.winfo_screenwidth() // 2) - (width // 2)
            y = (result_window.winfo_screenheight() // 2) - (height // 2)
            result_window.geometry('+{}+{}'.format(x, y))
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            extent = [0, n_points, n_frames, 0]
            
            im = ax.imshow(contraction_sizes, aspect='auto', cmap='gray_r', 
                        vmin=0, vmax=self.max_contraction.get(), extent=extent)

            if curvature_type == "greater":
                ax.invert_xaxis()
            
            ax.set_xticks(np.arange(0, n_points, 20))
            ax.set_title('Contraction Size Over Time')
            ax.set_xlabel('Point Number')
            ax.set_ylabel('Frame Number')
            plt.colorbar(im, ax=ax, label='Contraction Size (mm)')
            
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=result_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
            button_frame = tk.Frame(result_window)
            button_frame.pack(pady=10)
            
            def save_figure():
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[
                        ("PNG files", "*.png"),
                        ("JPEG files", "*.jpg;*.jpeg"),
                        ("TIFF files", "*.tif;*.tiff"),
                        ("All files", "*.*")
                    ],
                    title=f"Save {curvature_type.capitalize()} Curvature Contraction Plot"
                )
                if file_path:
                    try:
                        file_ext = os.path.splitext(file_path)[1].lower()
                        supported_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
                        
                        if not file_ext or file_ext not in supported_extensions:
                            file_path = os.path.splitext(file_path)[0] + '.png'
                            file_ext = '.png'
                        
                        if file_ext in ['.jpg', '.jpeg']:
                            format = 'jpeg'
                        elif file_ext == '.tif' or file_ext == '.tiff':
                            format = 'tiff'
                        else:  
                            format = 'png'
                        
                        fig.savefig(file_path, dpi=OUTPUT_DPI, bbox_inches='tight', format=format)
                        
                        messagebox.showinfo("Success", f"Figure saved to:\n{file_path}")
                        
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to save figure: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
            save_image_button = tk.Button(button_frame, text="Save Figure", command=save_figure, width=12)
            save_image_button.pack(side=tk.LEFT, padx=5)

            close_button = tk.Button(button_frame, text="Close", command=result_window.destroy, width=12)
            close_button.pack(side=tk.LEFT, padx=5)
        
            result_window.protocol("WM_DELETE_WINDOW", result_window.destroy)
                
        except Exception as e:
                messagebox.showerror("Error", f"Visualization failed: {str(e)}")
                import traceback
                traceback.print_exc()

    # def visualize_occlusion_results(percentage_occlusion, curvature_type="lesser"):
    #     """
    #     Visualize percentage occlusion results with similar style to contraction visualization.
        
    #     Args:
    #         percentage_occlusion: Percentage occlusion matrix (time × point)
    #         curvature_type: "lesser" or "greater"
    #     """
    #     n_frames = percentage_occlusion.shape[0]
    #     n_points = percentage_occlusion.shape[1]
        
    #     plt.figure(figsize=(10, 4))
    #     extent = [0, n_points, n_frames, 0]  # y-axis is frame number

    #     # 使用与contraction相同的可视化风格，但不同的颜色映射
    #     im = plt.imshow(percentage_occlusion, aspect='auto', cmap='gray_r', 
    #                 vmin=0, vmax=100, extent=extent)  # 百分比范围0-100

    #     if curvature_type == "greater":
    #         plt.gca().invert_xaxis()
        
    #     plt.xticks(np.arange(0, n_points, 20))
    #     plt.title('Percentage Occlusion Over Time')
    #     plt.xlabel('Point Number')
    #     plt.ylabel('Frame Number')
    #     plt.colorbar(im, label='Percentage Occlusion (%)')
        
    #     plt.tight_layout()
    #     plt.savefig(f"{curvature_type}_curvature_occlusion_analysis.png", dpi=OUTPUT_DPI)
    #     plt.show()

    def visualize_tracked_waves(self, contraction_sizes, wave_points, region_boundaries, intersections=None, 
                                wave_interpolated_points=None, curvature_type="lesser", max_contraction=10):
        """
        Visualize tracked waves on contraction map with scatter points (embedded in Tkinter).
        """
        try:
            n_frames, n_points = contraction_sizes.shape
            
            result_window = tk.Toplevel(self.root)
            result_window.title(f"{curvature_type.capitalize()} Curvature Wave Tracking Results")
            result_window.geometry("1300x800")
            
            result_window.transient(self.root)
            result_window.grab_set()
            result_window.lift()
            result_window.focus_force()
            
            fig_frame = tk.Frame(result_window)
            fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            canvas = FigureCanvasTkAgg(fig, master=fig_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            
            # Create base image
            im = ax.imshow(
                contraction_sizes,aspect='auto',cmap='gray_r',vmin=0,vmax=max_contraction,extent=[0, n_points, n_frames, 0],zorder=0
            )
            
            # Plot region boundaries
            for boundary in region_boundaries:
                ax.axvline(x=boundary, color='blue', linestyle='-', alpha=0.5, zorder=1)
            
            # Highlight excluded regions
            ax.axvspan(region_boundaries[0], region_boundaries[1], color='red', alpha=0.2, zorder=1)
            ax.axvspan(region_boundaries[-2], region_boundaries[-1], color='red', alpha=0.2, zorder=1)
            
            # Plot wave points using pre-calculated interpolated points
            if wave_interpolated_points:
                for wave_id, wave_data in wave_interpolated_points.items():
                    # Plot internal points (red points)
                    for region_key, points in wave_data['internal_points'].items():
                        if points:
                            region_times = [t for t, _ in points]
                            region_positions = [p for _, p in points]
                            
                            ax.scatter(region_positions, region_times,color='red', s=10, marker='o', alpha=1.0, zorder=2)
                    
                    # Plot boundary points (blue points)
                    boundary_points = wave_data['boundary_points']
                    if boundary_points:
                        t_inters = [t for t, _ in boundary_points]
                        p_inters = [p for _, p in boundary_points]
                        
                        ax.scatter(p_inters, t_inters,color='blue', s=25, marker='o',alpha=1.0, zorder=4)
            # Fallback to old method if pre-calculated points not available
            elif wave_points:
                for i, wave in enumerate(wave_points):
                    if not wave:
                        continue
                        
                    t_coords = [t for t, _ in wave]
                    p_coords = [p for _, p in wave]

                    # Create interpolation if enough points
                    if len(wave) > 2:
                        try:
                            # Sort points by position
                            sorted_indices = np.argsort(p_coords)
                            p_sorted = np.array(p_coords)[sorted_indices]
                            t_sorted = np.array(t_coords)[sorted_indices]
                            
                            # Get min and max of the actual data points
                            min_p, max_p = min(p_sorted), max(p_sorted)
                            
                            # Linear interpolation function
                            f = interpolate.interp1d(p_sorted, t_sorted, kind='linear',bounds_error=False, fill_value=np.nan)
                            
                            # Create 9 evenly distributed points within each region
                            for r in range(len(region_boundaries) - 1):
                                region_start = region_boundaries[r]
                                region_end = region_boundaries[r + 1]
                                
                                # Skip excluded regions
                                if region_start < region_boundaries[1] or region_end > region_boundaries[-2]:
                                    continue
                                
                                # Check if wave passes through this region
                                if max_p < region_start or min_p > region_end:
                                    continue
                                
                                # Generate 9 evenly spaced points within this region
                                region_positions = np.linspace(region_start, region_end, 11)[1:-1]
                                
                                # Only keep points that are within the actual data range
                                valid_mask = (region_positions >= min_p) & (region_positions <= max_p)
                                valid_positions = region_positions[valid_mask]
                                
                                if len(valid_positions) == 0:
                                    continue
                                
                                # Generate corresponding times using the interpolation function
                                region_times = f(valid_positions)
                                
                                # Remove any NaN values
                                valid_mask = ~np.isnan(region_times)
                                valid_positions = valid_positions[valid_mask]
                                region_times = region_times[valid_mask]
                                
                                if len(valid_positions) == 0:
                                    continue
                                
                                # Plot these points
                                ax.scatter(valid_positions, region_times,color='red', s=10, marker='o',  alpha=1.0, zorder=2)
                                
                        except Exception as e:
                            print(f"Error creating interpolated points: {e}")
                    
                    # Plot intersections if provided
                    if intersections and i < len(intersections):
                        wave_intersects = intersections[i]
                        if wave_intersects:
                            t_inters = [t for t, _ in wave_intersects]
                            p_inters = [p for _, p in wave_intersects]
                            
                            ax.scatter(p_inters, t_inters,color='blue', s=25, marker='o',alpha=1.0, zorder=4)
            
            # Add labels and title
            ax.set_xticks(np.arange(0, n_points, 20))
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            ax.set_title(f'{curvature_type.capitalize()} Curvature Contraction Map with Wave Tracking', fontsize=16)
            ax.set_xlabel('Point Number', fontsize=16)
            ax.set_ylabel('Frame Number', fontsize=16)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Contraction Size [mm]', fontsize=14)
            cbar.ax.tick_params(labelsize=12)
            
            if curvature_type == "greater":
                ax.invert_xaxis()
            
            plt.tight_layout()
            canvas.draw()
            
            # 创建按钮框架
            button_frame = tk.Frame(result_window)
            button_frame.pack(pady=10)
            
            def save_figure():
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"),("JPEG files", "*.jpg;*.jpeg"),("TIFF files", "*.tif;*.tiff"), ("All files", "*.*")],
                    title=f"Save {curvature_type.capitalize()} Curvature Wave Tracking Plot"
                )
                if file_path:
                    try:
                        file_ext = os.path.splitext(file_path)[1].lower()
                        supported_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
                        
                        if not file_ext or file_ext not in supported_extensions:
                            file_path = os.path.splitext(file_path)[0] + '.png'
                            file_ext = '.png'
                        
                        if file_ext in ['.jpg', '.jpeg']:
                            format = 'jpeg'
                        elif file_ext == '.tif' or file_ext == '.tiff':
                            format = 'tiff'
                        else:  
                            format = 'png'

                        fig.savefig(file_path, dpi=OUTPUT_DPI, bbox_inches='tight', format=format)
                        messagebox.showinfo("Success", f"Figure saved to {file_path}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to save figure: {str(e)}")
            
            save_button = tk.Button(button_frame, text="Save Figure", command=save_figure)
            close_button = tk.Button(button_frame, text="Close", command=result_window.destroy)
            
            save_button.pack(side=tk.LEFT, padx=10)
            close_button.pack(side=tk.LEFT, padx=10)
            
            result_window.protocol("WM_DELETE_WINDOW", result_window.destroy)
            
        except Exception as e:
            messagebox.showerror("Error", f"Wave tracking visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def manual_wave_tracking(self):
        if not hasattr(self, 'contraction_data') or self.contraction_data is None:
            messagebox.showwarning("Warning", "Please run contraction analysis first")
            return
        
        try:
            selection_window = tk.Toplevel(self.root)
            selection_window.title("Select Curvature for Wave Tracking")
            selection_window.geometry("350x180")
            
            selection_window.transient(self.root)
            selection_window.grab_set()
            selection_window.lift()
            selection_window.focus_force()
            
            selection_window.update_idletasks()
            width = selection_window.winfo_width()
            height = selection_window.winfo_height()
            x = (selection_window.winfo_screenwidth() - width) // 2
            y = (selection_window.winfo_screenheight() - height) // 2
            selection_window.geometry(f"+{x}+{y}")
            
            label = tk.Label(selection_window, 
                            text="Select curvature for wave tracking:", 
                            font=("Arial", 12))
            label.pack(pady=15)
            
            current_curvature = self.contraction_data.get('curvature_type', 'unknown')
            info_label = tk.Label(selection_window,
                                text=f"Current analyzed data: {current_curvature.capitalize()} Curvature",
                                font=("Arial", 10),
                                fg="blue")
            info_label.pack(pady=5)
            
            button_frame = tk.Frame(selection_window)
            button_frame.pack(pady=10)
            
            def track_lesser_curvature():
                selection_window.destroy()
                self._track_waves_for_curvature("lesser")
            
            def track_greater_curvature():
                selection_window.destroy()
                self._track_waves_for_curvature("greater")
            
            lesser_button = tk.Button(button_frame, text="Track Lesser Curvature", 
                                    command=track_lesser_curvature, width=20, height=2)
            greater_button = tk.Button(button_frame, text="Track Greater Curvature", 
                                    command=track_greater_curvature, width=20, height=2)
            
            lesser_button.pack(side=tk.LEFT, padx=10)
            greater_button.pack(side=tk.LEFT, padx=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Wave tracking selection failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def _track_waves_for_curvature(self, curvature_type):
        try:
            if (not hasattr(self, 'contraction_data') or 
                self.contraction_data is None or 
                self.contraction_data.get('curvature_type') != curvature_type):
                
                result = messagebox.askyesno(
                    "Analysis Required", 
                    f"{curvature_type.capitalize()} curvature analysis is required for wave tracking.\n"
                    f"Do you want to run {curvature_type} curvature analysis now?"
                )
                
                if result:
                    self.analyze_contraction(curvature_type)
                    if (hasattr(self, 'contraction_data') and 
                        self.contraction_data is not None and 
                        self.contraction_data.get('curvature_type') == curvature_type):
                        pass
                    else:
                        return
                else:
                    return
            
            cont_sizes = self.contraction_data['contraction_sizes']
            
            if curvature_type == "lesser":
                spacing = self.lesser_spacing.get()
            else:
                spacing = self.greater_spacing.get()
            
            frame_duration = self.frame_duration.get()
            max_contraction = self.max_contraction.get()
            
            wave_points, physical_speeds, region_boundaries, intersections, interpolated_points = self.manually_track_waves(
                cont_sizes, spacing, frame_duration, N_REGIONS, curvature_type, max_contraction)
            
            self.visualize_tracked_waves(cont_sizes, wave_points, region_boundaries, 
                                    intersections, interpolated_points, curvature_type, max_contraction)
            
            export_wave_data(interpolated_points, region_boundaries, spacing, frame_duration, curvature_type)
            export_region_wave_contraction_stats(cont_sizes, interpolated_points, region_boundaries, curvature_type)
            
            messagebox.showinfo("Success", f"{curvature_type.capitalize()} curvature wave tracking completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"{curvature_type.capitalize()} curvature wave tracking failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def manually_track_waves(self, contraction_sizes, point_spacing_mm, frame_duration_s, n_regions, curvature_type="lesser", max_contraction=10):
        """
        Interactive function to manually select wave points and calculate speeds.
        """
        n_frames, n_points = contraction_sizes.shape
        
        # Create regions
        region_boundaries = np.linspace(0, n_points, n_regions+1, dtype=int)
        
        tracking_window = tk.Toplevel(self.root)
        tracking_window.title(f"{curvature_type.capitalize()} Curvature Wave Tracking")
        tracking_window.geometry("1200x700")

        tracking_window.transient(self.root)
        tracking_window.grab_set()
        tracking_window.lift()
        tracking_window.focus_force()
        
        fig_frame = tk.Frame(tracking_window)
        fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        fig, ax = plt.subplots(figsize=(12, 6))
        canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        extent = [0, n_points, n_frames, 0]
        im = ax.imshow(contraction_sizes, aspect='auto', cmap='gray_r', vmin=0, vmax=max_contraction, extent=extent)
        
        cbar = fig.colorbar(im, ax=ax, label='Contraction Size (mm)')
        cbar.set_label('Contraction Size (mm)', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        
        # Plot region boundaries
        for boundary in region_boundaries:
            ax.axvline(x=boundary, color='blue', linestyle='-', alpha=0.5)
        
        # Highlight excluded regions
        ax.axvspan(region_boundaries[0], region_boundaries[1], color='red', alpha=0.2)
        ax.axvspan(region_boundaries[-2], region_boundaries[-1], color='red', alpha=0.2)
        
        # Add labels
        ax.set_xticks(np.arange(0, n_points, 20))
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_title(f'{curvature_type.capitalize()} Curvature Contraction Map - Click to Track Waves', fontsize=16)
        ax.set_xlabel('Point Number', fontsize=16)
        ax.set_ylabel('Frame Number', fontsize=16)
        
        if curvature_type == "greater":
            ax.invert_xaxis()
        
        plt.tight_layout()
        canvas.draw()
        
        # Tracking variables
        all_waves = []
        current_wave = []
        wave_line = None
        wave_scatter = None
        
        button_frame = tk.Frame(tracking_window)
        button_frame.pack(pady=10)
        
        info_var = tk.StringVar()
        info_var.set('Click to add points. "Z": undo last point. "N": new wave. "F": finish.')
        info_label = tk.Label(button_frame, textvariable=info_var, font=("Arial", 12))
        info_label.pack(pady=5)
        
        def new_wave():
            nonlocal current_wave
            if current_wave:
                current_wave.sort(key=lambda x: x[1])
                all_waves.append(current_wave.copy())
                current_wave = []
                update_wave_plot()
                info_var.set(f'Started new wave ({len(all_waves)} complete). Click to add points.')
        
        def finish_tracking():
            nonlocal current_wave
            if current_wave:
                current_wave.sort(key=lambda x: x[1])
                all_waves.append(current_wave.copy())
            tracking_window.destroy()
        
        def undo_last():
            nonlocal current_wave
            if current_wave:
                current_wave.pop()
                update_wave_plot()
                if not current_wave:
                    info_var.set('All points removed. Click to add points.')
                else:
                    info_var.set(f'Current wave: {len(current_wave)} points.')
        
        new_button = tk.Button(button_frame, text="New Wave (N)", command=new_wave)
        undo_button = tk.Button(button_frame, text="Undo Last (Z)", command=undo_last)
        finish_button = tk.Button(button_frame, text="Finish (F)", command=finish_tracking)
        
        new_button.pack(side=tk.LEFT, padx=5)
        undo_button.pack(side=tk.LEFT, padx=5)
        finish_button.pack(side=tk.LEFT, padx=5)

        def update_wave_plot():
            """Update the plot with current wave points"""
            nonlocal wave_line, wave_scatter
            
            # Clear previous artists
            for artist in [wave_line, wave_scatter]:
                if artist is not None and artist in ax.get_children():
                    artist.remove()
            
            if current_wave:
                # Extract coordinates
                p_coords = [p for _, p in current_wave]
                t_coords = [t for t, _ in current_wave]
                
                # Create new scatter plot
                wave_scatter = ax.scatter(p_coords, t_coords, color='red', 
                                        s=25, marker='o', zorder=3)
                
                # Create line if enough points
                if len(current_wave) > 1:
                    sorted_wave = sorted(current_wave, key=lambda x: x[1])
                    p_coords = [p for _, p in sorted_wave]
                    t_coords = [t for t, _ in sorted_wave]
                    wave_line = ax.plot(p_coords, t_coords, 'r-', 
                                    linewidth=1.5, zorder=2)[0]
            
            canvas.draw()
            info_var.set(f'Current wave: {len(current_wave)} points. "Z": undo last point. "N": new wave. "F": finish.')

        def on_click(event):
            """Handle click events"""
            if event.inaxes != ax:
                return
            
            current_wave.append((event.ydata, event.xdata))  # (frame, position)
            update_wave_plot()

        def on_key(event):
            """Handle key press events"""
            if event.key == 'n' or event.key == 'N':  # New wave
                new_wave()
            elif event.key == 'f' or event.key == 'F':  # Finish
                finish_tracking()
            elif event.key == 'z' or event.key == 'Z':  # Undo last point
                undo_last()
        
        # Connect event handlers
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        tracking_window.wait_window()
        
        wave_boundary_intersections = []
        wave_interpolated_points = {}
        
        for wave_id, wave in enumerate(all_waves):
            wave_interpolated_points[wave_id] = {
                'boundary_points': [],
                'internal_points': {}
            }
            
            if len(wave) < 3:
                wave_boundary_intersections.append([])
                continue
                
            wave.sort(key=lambda x: x[1])
            t_coords = [t for t, _ in wave]
            p_coords = [p for _, p in wave]
            
            min_p, max_p = min(p_coords), max(p_coords)
            
            try:
                f = interpolate.interp1d(p_coords, t_coords, kind='linear', 
                                        bounds_error=False, fill_value="extrapolate")
                
                intersections = []
                for boundary in region_boundaries:
                    if min_p <= boundary <= max_p:
                        boundary_time = f(boundary)
                        intersections.append((boundary_time, boundary))
                        wave_interpolated_points[wave_id]['boundary_points'].append((boundary_time, boundary))
                        
                wave_boundary_intersections.append(intersections)
                
                first_boundary = region_boundaries[1]
                last_boundary = region_boundaries[-2]
                
                for r in range(len(region_boundaries) - 1):
                    region_start = region_boundaries[r]
                    region_end = region_boundaries[r + 1]
                    
                    if region_start < first_boundary or region_end > last_boundary:
                        continue
                    
                    if max_p < region_start or min_p > region_end:
                        continue
                    
                    region_positions = np.linspace(region_start, region_end, 11)[1:-1]
                    valid_mask = (region_positions >= min_p) & (region_positions <= max_p)
                    valid_positions = region_positions[valid_mask]
                    
                    if len(valid_positions) == 0:
                        continue
                    
                    region_times = f(valid_positions)
                    valid_mask = ~np.isnan(region_times)
                    valid_positions = valid_positions[valid_mask]
                    region_times = region_times[valid_mask]
                    
                    if len(valid_positions) == 0:
                        continue
                    
                    region_key = (region_start, region_end)
                    wave_interpolated_points[wave_id]['internal_points'][region_key] = list(zip(region_times, valid_positions))
                    
            except Exception as e:
                print(f"Linear interpolation failed: {e}")
                wave_boundary_intersections.append([])
        
        # Calculate physical speeds
        physical_speeds = []
        for intersections in wave_boundary_intersections:
            wave_speeds = []
            for i in range(len(intersections) - 1):
                t1, p1 = intersections[i]
                t2, p2 = intersections[i + 1]
                if t2 != t1:
                    speed_mm_s = (p2 - p1) * point_spacing_mm / abs((t2 - t1) * frame_duration_s)
                    wave_speeds.append(speed_mm_s)
            physical_speeds.append(wave_speeds)
        
        return all_waves, physical_speeds, region_boundaries, wave_boundary_intersections, wave_interpolated_points


    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Display frame for image and plotting
        self.display_frame = tk.Frame(main_frame)
        self.display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add toolbar to the bottom of display area
        toolbar_frame = tk.Frame(self.display_frame, height=80)  
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0,3)) 

        status_container = tk.Frame(toolbar_frame, bd=1, relief=tk.SUNKEN)
        status_container.pack(side=tk.LEFT, fill=tk.X, expand=False, padx=(0,3))

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(
            status_container, textvariable=self.status_var,anchor=tk.W,padx=5,pady=2,       
            bg='#f0f0f0',font=('Arial', 9), height=2,width=100)
        self.status_label.pack(fill=tk.X, expand=True)

        # Matplotlib figure for displaying the image
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.display_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Control panel
        control_panel = tk.Frame(main_frame, width=200)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Mode indicators
        mode_frame = tk.LabelFrame(control_panel, text="Mode")
        mode_frame.pack(fill=tk.X, pady=5)

        self.mode_var = tk.StringVar(value="View")
        self.mode_label = tk.Label(mode_frame, textvariable=self.mode_var, font=("Arial", 12, "bold"))
        self.mode_label.pack(pady=5)

        # Navigation
        nav_frame = tk.LabelFrame(control_panel, text="Navigation")
        nav_frame.pack(fill=tk.X, pady=5)
        nav_buttons_frame = tk.Frame(nav_frame)
        nav_buttons_frame.pack(fill=tk.X)  

        self.prev_button = tk.Button(nav_buttons_frame, text="Previous", command=self.prev_image, state=tk.DISABLED)
        self.next_button = tk.Button(nav_buttons_frame, text="Next", command=self.next_image, state=tk.DISABLED)

        self.prev_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.next_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew")

        nav_buttons_frame.columnconfigure(0, weight=1)
        nav_buttons_frame.columnconfigure(1, weight=1)
                
        jump_controls_frame = tk.Frame(nav_frame)
        jump_controls_frame.pack(fill=tk.X)  

        self.jump_entry = tk.Entry(jump_controls_frame)
        self.jump_button = tk.Button(jump_controls_frame, text="Jump", command=self.jump_to_frame)

        self.jump_entry.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        self.jump_button.grid(row=1, column=1, padx=2, pady=2, sticky="ew")

        jump_controls_frame.columnconfigure(0, weight=3)  
        jump_controls_frame.columnconfigure(1, weight=1)  
    
        zoom_frame = tk.LabelFrame(control_panel, text="Zoom Controls")
        zoom_frame.pack(fill=tk.X, pady=5)

        zoom_buttons_frame = tk.Frame(zoom_frame)
        zoom_buttons_frame.pack(fill=tk.X)  

        self.zoom_in_button = tk.Button(zoom_buttons_frame, text="Zoom In", command=lambda: self.zoom_image(1))
        self.zoom_out_button = tk.Button(zoom_buttons_frame, text="Zoom Out", command=lambda: self.zoom_image(-1))

        self.zoom_in_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.zoom_out_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew")

        zoom_buttons_frame.columnconfigure(0, weight=1)
        zoom_buttons_frame.columnconfigure(1, weight=1)

        self.reset_zoom_button = tk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom)
        self.reset_zoom_button.pack(fill=tk.X, pady=2)

        self.counter_var = tk.StringVar(value="Image: 0/0")

        # ROI controls
        roi_frame = tk.LabelFrame(control_panel, text="ROI Controls")
        roi_frame.pack(fill=tk.X, pady=5)

        self.draw_button = tk.Button(roi_frame, text="Draw ROI", command=self.start_drawing_roi)
        self.draw_button.pack(fill=tk.X, pady=2)

        self.complete_button = tk.Button(roi_frame, text="Complete ROI", command=self.complete_roi, state=tk.DISABLED)
        self.complete_button.pack(fill=tk.X, pady=2)

        self.edit_button = tk.Button(roi_frame, text="Edit ROI", command=self.toggle_edit_mode, state=tk.DISABLED)
        self.edit_button.pack(fill=tk.X, pady=2)

        clear_copy_buttons_frame = tk.Frame(roi_frame)
        clear_copy_buttons_frame.pack(fill=tk.X)  
        self.clear_button = tk.Button(clear_copy_buttons_frame, text="Clear ROI", command=self.clear_roi, state=tk.DISABLED)
        self.copy_button = tk.Button(clear_copy_buttons_frame, text="Copy to Next", command=self.copy_roi_to_next, state=tk.DISABLED)
        self.clear_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.copy_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        clear_copy_buttons_frame.columnconfigure(0, weight=1)
        clear_copy_buttons_frame.columnconfigure(1, weight=1)

        
        self.toggle_control_button = tk.Button(roi_frame, text="Hide Control Points", command=self.toggle_control_points,state=tk.DISABLED)
        self.toggle_control_button.pack(fill=tk.X, pady=2)

        Mark_buttons_frame = tk.Frame(roi_frame)
        Mark_buttons_frame.pack(fill=tk.X)  
        self.mark_major_button = tk.Button(Mark_buttons_frame, text="Proximal Point", command=lambda: self.set_keypoint_type('major'))
        self.mark_minor_button = tk.Button(Mark_buttons_frame, text="Distal Point", command=lambda: self.set_keypoint_type('minor'))
        self.mark_major_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.mark_minor_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        Mark_buttons_frame.columnconfigure(0, weight=1)
        Mark_buttons_frame.columnconfigure(1, weight=1)

        self.clear_keys_button = tk.Button(roi_frame, text="Clear Key Points", command=self.clear_key_points)
        self.clear_keys_button.pack(fill=tk.X, pady=2)

        # Add propagation controls
        Propagate_buttons_frame = tk.Frame(roi_frame)
        Propagate_buttons_frame.pack(fill=tk.X)  
        self.propagate_button = tk.Button(Propagate_buttons_frame, text="Propagate to Next Frame",command=self.propagate_roi_with_flow)
        self.propagate_button.pack(fill=tk.X, pady=2)

        # Centerline frame -
        centerline_frame = tk.LabelFrame(control_panel, text="Centerline")
        centerline_frame.pack(fill=tk.X, pady=5)
        
        Split_buttons_frame = tk.Frame(centerline_frame)
        Split_buttons_frame.pack(fill=tk.X)  
        self.split_button = tk.Button(Split_buttons_frame, text="Split Current", command=self.split_at_key_points, state=tk.DISABLED)
        self.split_all_button = tk.Button(Split_buttons_frame, text="Split All", command=self.split_all_frames, state=tk.DISABLED)

        self.split_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.split_all_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        Split_buttons_frame.columnconfigure(0, weight=1)
        Split_buttons_frame.columnconfigure(1, weight=1)

        self.toggle_split_button = tk.Button(centerline_frame, text="Hide Split Points", command=self.toggle_split_points,state=tk.DISABLED)
        self.toggle_split_button.pack(fill=tk.X, pady=2)

        Calculate_buttons_frame = tk.Frame(centerline_frame)
        Calculate_buttons_frame.pack(fill=tk.X)  
        self.calc_midpoints_button = tk.Button(Calculate_buttons_frame, text="Calc Midpoints", command=self.calculate_current_midpoints, state=tk.DISABLED)
        self.calc_all_midpoints_button = tk.Button(Calculate_buttons_frame, text="Calc All Mid", command=self.calculate_all_midpoints, state=tk.DISABLED)
        self.calc_midpoints_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.calc_all_midpoints_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        Calculate_buttons_frame.columnconfigure(0, weight=1)
        Calculate_buttons_frame.columnconfigure(1, weight=1)

        self.toggle_midpoints_button = tk.Button(centerline_frame, text="Hide Midpoints", command=self.toggle_midpoints, state=tk.DISABLED)
        self.toggle_midpoints_button.pack(fill=tk.X, pady=2)

        self.fit_button = tk.Button(centerline_frame, text="Fit Centerline", command=self.fit_polynomial_to_midpoints)
        self.fit_button.pack(fill=tk.X, pady=2)
        self.centerline_toggle_button = tk.Button(centerline_frame, text="Toggle Centerline", command=self.toggle_centerline)
        self.centerline_toggle_button.pack(fill=tk.X, pady=2)

        # Image enhancement controls
        enhancement_frame = tk.LabelFrame(control_panel, text="Image Enhancement")
        enhancement_frame.pack(fill=tk.X, pady=5)

        # Brightness adjustment
        brightness_label_frame = tk.Frame(enhancement_frame)
        brightness_label_frame.pack(fill=tk.X)
        tk.Label(brightness_label_frame, text="Brightness:").pack(side=tk.LEFT, padx=5)
        self.brightness_value_label = tk.Label(brightness_label_frame, text="1.0")
        self.brightness_value_label.pack(side=tk.RIGHT, padx=5)

        self.brightness_scale_var = tk.DoubleVar(value=1.0)
        self.brightness_scale = tk.Scale(enhancement_frame, from_=0.1, to=3.0, resolution=0.1, 
                                        orient=tk.HORIZONTAL, variable=self.brightness_scale_var, 
                                        command=self.adjust_brightness)
        self.brightness_scale.pack(fill=tk.X, pady=(0, 5))

        # Contrast adjustment
        contrast_label_frame = tk.Frame(enhancement_frame)
        contrast_label_frame.pack(fill=tk.X)
        tk.Label(contrast_label_frame, text="Contrast:").pack(side=tk.LEFT, padx=5)
        self.contrast_value_label = tk.Label(contrast_label_frame, text="1.0")
        self.contrast_value_label.pack(side=tk.RIGHT, padx=5)

        self.contrast_scale_var = tk.DoubleVar(value=1.0)
        self.contrast_scale = tk.Scale(enhancement_frame, from_=0.1, to=3.0, resolution=0.1, 
                                    orient=tk.HORIZONTAL, variable=self.contrast_scale_var, 
                                    command=self.adjust_contrast)
        self.contrast_scale.pack(fill=tk.X, pady=(0, 5))

        # Reset button
        self.reset_button = tk.Button(enhancement_frame, text="Reset to Default", command=self.reset_enhancement)
        self.reset_button.pack(fill=tk.X, pady=2)

        # Advanced enhancement options
        advanced_frame = tk.Frame(enhancement_frame)
        advanced_frame.pack(fill=tk.X, pady=5)

        self.enhancement_method_var = tk.StringVar(value="Manual")
        tk.Label(advanced_frame, text="Enhancement Method:").pack(side=tk.LEFT, padx=5)
        self.enhancement_method = ttk.Combobox(advanced_frame, textvariable=self.enhancement_method_var, 
                                            values=["Manual", "Histogram Equalization", "CLAHE"],state="readonly", width=15)
        self.enhancement_method.pack(side=tk.RIGHT, padx=5)
        self.enhancement_method.bind("<<ComboboxSelected>>", self.change_enhancement_method)

        # Optical Flow controls 
        flow_frame = tk.LabelFrame(control_panel, text="Optical Flow")
        flow_frame.pack(fill=tk.X, pady=5)

        self.flow_button = tk.Button(flow_frame, text="Toggle Optical Flow", command=self.toggle_optical_flow)
        self.flow_button.pack(fill=tk.X, pady=2)

    def setup_interactions(self):
        # Mouse events for ROI drawing and editing
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Keyboard navigation
        self.root.bind('<Left>', lambda event: self.prev_image())
        self.root.bind('<Right>', lambda event: self.next_image())

        # Bind Ctrl+Z to undo
        self.root.bind('<Control-z>', lambda event: self.undo())
        self.root.bind('<Return>', lambda event: self.jump_to_frame())

        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.root.bind('<Control-s>', lambda event: self.complete_roi())
        self.root.bind('<Control-e>', lambda event: self.toggle_edit_mode())
        #self.root.bind('<Control-x>', lambda event: self.propagate_roi_with_flow())
        self.root.bind('<Control-x>', lambda event: self.propagate_roi_with_flow() if self.current_index < len(self.image_files) - 1 and self.current_index in self.roi_points else messagebox.showwarning("Warning", "No next frame or ROI to propagate"))

    def load_image_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder Containing MRI Images")
        if folder_path:
            self.image_folder = folder_path
            self.load_images()

    def load_images(self):
        self.image_files = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

        try:
            # Get all image files
            for file in sorted(os.listdir(self.image_folder)):
                if file.lower().endswith(valid_extensions):
                    self.image_files.append(os.path.join(self.image_folder, file))

            if not self.image_files:
                messagebox.showerror("Error", "No valid image files found in the selected folder")
                return

            # If workspace was loaded, keep the ROI points
            if not hasattr(self, 'roi_points') or self.roi_points is None:
                self.roi_points = {}

            # Reset current index if not set by workspace load
            if self.current_index >= len(self.image_files):
                self.current_index = 0

            # Update UI
            self.update_counter()
            self.display_current_image()

            # Enable buttons
            self.prev_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.NORMAL)
            
            self.status_var.set(
                "Ctrl+E (Edit ROI), Ctrl+P (Propagate ROI), Ctrl+S (Complete ROI), Ctrl+Z (Undo)"
            )

            # Show success message
            messagebox.showinfo("Success", f"Loaded {len(self.image_files)} images successfully")

            # Show Shortcuts message
            messagebox.showinfo("Tips", f"Shortcuts are showed in the Status Bar")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading images: {str(e)}")

    def display_current_image(self):
        if 0 <= self.current_index < len(self.image_files):
            # Clear the current axis
            self.ax.clear()
            
            # Load and display the image
            img = Image.open(self.image_files[self.current_index])
            img_array = np.array(img)
            
            # Store current image dimensions
            self.current_image = img_array
            h, w = img_array.shape[:2] if len(img_array.shape) > 2 else (img_array.shape[0], img_array.shape[1])
            
            # Adjust brightness
            img_array = self.enhance_image(img_array)
            
            # Check if image is grayscale or color
            if len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1):
                self.ax.imshow(img_array, cmap='gray')
            else:
                self.ax.imshow(img_array)
            
            self.ax.set_title(f"Image {self.current_index + 1}/{len(self.image_files)}")
            self.ax.axis('off')
            
            # Set fixed display range based on image dimensions
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)  # Inverted y-axis for image coordinates
            self.ax.autoscale(False)  # Disable autoscaling
            
            # Store these limits
            self.xlim = self.ax.get_xlim()
            self.ylim = self.ax.get_ylim()
            
            # Reset zoom scale
            self.zoom_scale = self.default_zoom
            
            # Reset ROI display elements
            self.control_point_markers = []
            self.spline_line = None
            
            # Load ROI for this frame if it exists
            if self.current_index in self.roi_points and len(self.roi_points[self.current_index]) > 0:
                self.control_points = self.roi_points[self.current_index].tolist()
                self.update_roi_display()
            else:
                self.control_points = []
            
            if self.current_index in self.spline_points_per_frame:
                self.spline_points = self.spline_points_per_frame[self.current_index]
                self.update_roi_display()

            if hasattr(self, 'centerline_data'):
                self.visualize_centerline()  

            # Update button states first
            self.update_button_states()
            
            # If optical flow display flag is True, then display optical flow
            # This needs to come after setting the axis limits
            if hasattr(self, 'show_flow') and self.show_flow:
                self.display_optical_flow()
            
            self.canvas.draw()

    def next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            # Exit drawing/editing mode if active
            if self.is_drawing or self.is_editing:
                self.complete_roi()

            # Move to the next frame
            self.current_index += 1
            self.update_counter()
            self.display_current_image()

            # If optical flow is enabled, display it for the new frame
            if self.show_flow:
                self.display_optical_flow()

    def prev_image(self):
        if self.image_files and self.current_index > 0:
            # Exit drawing/editing mode if active
            if self.is_drawing or self.is_editing:
                self.complete_roi()

            # Move to the previous frame
            self.current_index -= 1
            self.update_counter()
            self.display_current_image()

            # If optical flow is enabled, display it for the new frame
            if self.show_flow:
                self.display_optical_flow()

    def jump_to_frame(self):
        if not self.image_files:
            messagebox.showwarning("Warning", "No images loaded")
            return

        try:
            frame_number = int(self.jump_entry.get())
            if 1 <= frame_number <= len(self.image_files):
                self.current_index = frame_number - 1
                self.display_current_image()
                self.update_counter()
                self.status_var.set(f"Jumped to frame {frame_number}")
            else:
                messagebox.showwarning("Warning", f"Frame number must be between 1 and {len(self.image_files)}")
        except ValueError:
            messagebox.showwarning("Warning", "Please enter a valid frame number")

    def on_scroll(self, event):
        if not event.inaxes:  
            return

        if event.button == 'up':
            self.zoom_scale += self.zoom_step
        elif event.button == 'down':
            self.zoom_scale -= self.zoom_step

        if self.zoom_scale < 0.1:
            self.zoom_scale = 0.1

        self.update_zoom(event.xdata, event.ydata)

    def update_zoom(self, center_x, center_y):
        if self.xlim is None or self.ylim is None:
            return

        width = (self.xlim[1] - self.xlim[0]) / self.zoom_scale
        height = (self.ylim[1] - self.ylim[0]) / self.zoom_scale

        new_xlim = (center_x - width / 2, center_x + width / 2)
        new_ylim = (center_y - height / 2, center_y + height / 2)

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def zoom_image(self, direction):
        if direction == 1:  # 放大
            self.zoom_scale += self.zoom_step
        elif direction == -1:  # 缩小
            self.zoom_scale -= self.zoom_step

        if self.zoom_scale < 0.1:
            self.zoom_scale = 0.1

        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        center_x = (current_xlim[0] + current_xlim[1]) / 2
        center_y = (current_ylim[0] + current_ylim[1]) / 2

        self.update_zoom(center_x, center_y)

    def reset_zoom(self):
        self.zoom_scale = self.default_zoom
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.canvas.draw()

    def adjust_brightness(self, value):
        self.brightness_scale = float(value)
        self.brightness_value_label.config(text=f"{float(value):.1f}")
        self.display_current_image()

    def adjust_contrast(self, value):
        self.contrast_scale = float(value)
        self.contrast_value_label.config(text=f"{float(value):.1f}")
        self.display_current_image()

    def reset_enhancement(self):
        self.brightness_scale_var.set(1.0)
        self.contrast_scale_var.set(1.0)
        self.brightness_value_label.config(text="1.0")
        self.contrast_value_label.config(text="1.0")
        self.enhancement_method_var.set("Manual")
        self.display_current_image()
        self.status_var.set("Image enhancement reset to default")

    def change_enhancement_method(self, event=None):
        method = self.enhancement_method_var.get()
        
        # Enable or disable manual controls based on selected method
        if method == "Manual":
            self.brightness_scale.config(state=tk.NORMAL)
            self.contrast_scale.config(state=tk.NORMAL)
        else:
            # Disable manual controls for automatic methods
            self.brightness_scale.config(state=tk.DISABLED)
            self.contrast_scale.config(state=tk.DISABLED)
        
        self.display_current_image()
        self.status_var.set(f"Enhancement method changed to {method}")

    def enhance_image(self, img_array):
        """
        Apply selected enhancement method to the image array.
        """
        method = self.enhancement_method_var.get()
        
        # Convert to float for processing if needed
        if img_array.dtype != np.float32:
            img_array = img_array.astype(np.float32)
        
        # Apply selected enhancement method
        if method == "Manual":
            # Apply brightness and contrast adjustments
            brightness = self.brightness_scale_var.get()
            contrast = self.contrast_scale_var.get()
            
            # Apply brightness (multiply)
            adjusted = img_array * brightness
            
            # Apply contrast (centered at mean)
            if contrast != 1.0:
                mean_val = np.mean(adjusted)
                adjusted = (adjusted - mean_val) * contrast + mean_val
            
            # Clip to valid range and convert back
            adjusted = np.clip(adjusted, 0, 255)
            return adjusted.astype(np.uint8)
        
        elif method == "Histogram Equalization":
            # Convert to uint8 for histogram equalization
            if len(img_array.shape) == 2 or img_array.shape[2] == 1:
                # Grayscale image
                img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
                return cv2.equalizeHist(img_uint8)
            else:
                # Color image - equalize each channel separately
                img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
                img_yuv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        elif method == "CLAHE":
            # Apply Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            
            # Convert to uint8 for CLAHE
            img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
            
            if len(img_array.shape) == 2 or img_array.shape[2] == 1:
                # Grayscale image
                return clahe.apply(img_uint8)
            else:
                # Color image - apply CLAHE to luminance channel
                img_lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
                img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
                return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        # Default fallback
        return img_array

    # Replace the existing adjust_image_brightness method with this:
    def adjust_image_brightness(self, img_array, brightness_scale):
        # This is just a wrapper for the new enhance_image method
        if brightness_scale == 1.0 and self.contrast_scale_var.get() == 1.0:
            return img_array  # No adjustment needed
        
        # Store the old brightness value
        old_brightness = self.brightness_scale_var.get()
        
        # Temporarily set the brightness value to the passed scale
        self.brightness_scale_var.set(brightness_scale)
        
        # Apply the enhancement
        result = self.enhance_image(img_array)
        
        # Restore the original brightness value
        self.brightness_scale_var.set(old_brightness)
        
        return result
    
    def update_counter(self):
        total = len(self.image_files)
        current = self.current_index + 1 if self.image_files else 0
        self.counter_var.set(f"Image: {current}/{total}")

    def start_drawing_roi(self):
        if not self.image_files:
            return

        # Clear existing ROI for this frame
        self.clear_roi()

        # Enter drawing mode
        self.is_drawing = True
        self.is_editing = False
        self.mode_var.set("Drawing ROI")
        self.status_var.set("Press Ctrl+e to enter edit, Right-Click to add points, Right-click to delete points, Press Ctrl+S to finish edit.")

        # Update buttons
        self.draw_button.config(state=tk.DISABLED)
        self.complete_button.config(state=tk.NORMAL)
        self.edit_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.NORMAL)

    def complete_roi(self):
        if self.is_drawing and len(self.control_points) >= 3:
            # Save the current control points to this frame
            self.roi_points[self.current_index] = np.array(self.control_points)

            # Exit drawing mode
            self.is_drawing = False

            # Update buttons
            self.mode_var.set("View")
            self.status_var.set("ROI completed")
            self.draw_button.config(state=tk.NORMAL)
            self.complete_button.config(state=tk.DISABLED)
            self.edit_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL)
            self.copy_button.config(state=tk.NORMAL)
            self.propagate_button.config(state=tk.NORMAL)
            #self.auto_prop_button.config(state=tk.NORMAL)

        elif self.is_editing:
            # Save the current control points to this frame
            self.roi_points[self.current_index] = np.array(self.control_points)

            # Exit editing mode
            self.is_editing = False
            self.selected_point = None

            # Update buttons and status
            self.mode_var.set("View")
            self.status_var.set("ROI editing completed")
            self.draw_button.config(state=tk.NORMAL)
            self.complete_button.config(state=tk.DISABLED)
            self.edit_button.config(state=tk.NORMAL)
            self.copy_button.config(state=tk.NORMAL)
            self.propagate_button.config(state=tk.NORMAL)
            #self.auto_prop_button.config(state=tk.NORMAL)

        elif self.is_drawing and len(self.control_points) < 3:
            messagebox.showwarning("Warning", "Need at least 3 points to create a valid ROI")
            return

    def clear_roi(self):
        if self.current_index in self.roi_points:
            del self.roi_points[self.current_index]

        self.control_points = []
        self.control_point_markers = []
        self.spline_line = None
        self.spline_points = []
        self.selected_point = None
        self.is_drawing = False
        self.is_editing = False

        if self.current_index in self.spline_points_per_frame:
            del self.spline_points_per_frame[self.current_index]

        # Update display
        self.display_current_image()
        self.toggle_split_button.config(state=tk.DISABLED)  
        self.toggle_control_button.config(state=tk.DISABLED)  

        # Update mode and buttons
        self.mode_var.set("View")
        self.status_var.set("ROI cleared")
        self.draw_button.config(state=tk.NORMAL)
        self.complete_button.config(state=tk.DISABLED)
        self.edit_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        self.copy_button.config(state=tk.DISABLED)
        self.propagate_button.config(state=tk.DISABLED)
        #self.auto_prop_button.config(state=tk.DISABLED)
        self.calc_midpoints_button.config(state=tk.DISABLED)
        self.calc_all_midpoints_button.config(state=tk.DISABLED)
        self.update_button_states()

    def split_at_key_points(self, silent=False):
        """Split the spline curve into two parts based on key points,then evenly divide each curve into 198 segments."""
        try:
            # 1. check the points
            if not self.validate_key_points():
                if not silent:
                    messagebox.showwarning("Warning", "please mark key points")
                return False

            # 2. generate spline with 1000 points
            points = np.array(self.control_points)
            points = np.vstack([points, points[0]])  # 闭合曲线
            tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=1)
            u_new = np.linspace(0, 1, 1000)
            spline_points = np.array(splev(u_new, tck)).T

            # 3. get the position of key points
            key_data = self.key_points[self.current_index]
            major_pos = key_data['major']['position']
            minor_pos = key_data['minor']['position']

            # 4. Locate the point on the spline nearest to the key point
            def find_nearest_point_idx(target, points):
                distances = np.linalg.norm(points - target, axis=1)
                return np.argmin(distances)

            major_idx = find_nearest_point_idx(major_pos, spline_points)
            minor_idx = find_nearest_point_idx(minor_pos, spline_points)

            # 5. split two parts according key points 
            start, end = sorted([major_idx, minor_idx])
            if major_idx < minor_idx:
                segment1 = spline_points[major_idx:minor_idx+1]
                segment2 = np.vstack([spline_points[minor_idx:], spline_points[:major_idx+1]])
            else:
                segment1 = spline_points[minor_idx:major_idx+1]
                segment2 = np.vstack([spline_points[major_idx:], spline_points[:minor_idx+1]])
                # 交换segment1和segment2以确保segment1总是大弯点到小弯点
                segment1, segment2 = segment2, segment1

            # 6. 计算每条曲线的弧长并均分为199份
            def split_segment(segment, num_parts=199, pixel_spacing=None):
                # 计算累积弧长
                diffs = np.diff(segment, axis=0)
                
                if self.has_pixel_spacing and self.pixel_spacing is not None:
                    spacing_x, spacing_y = self.pixel_spacing
                    dx = diffs[:, 0] * spacing_x  # X方向的物理位移
                    dy = diffs[:, 1] * spacing_y  # Y方向的物理位移
                    dists = np.sqrt(dx**2 + dy**2)  # 物理距离
                else:
                    dists = np.linalg.norm(diffs, axis=1)

                cum_dists = np.insert(np.cumsum(dists), 0, 0)
                total_length = cum_dists[-1]
                
                # 计算分割点位置
                split_positions = np.linspace(0, total_length, num_parts+1)
                
                # 插值找到对应点
                split_points = []
                for pos in split_positions:
                    idx = np.searchsorted(cum_dists, pos) - 1
                    if idx < 0:
                        idx = 0
                    if idx >= len(segment)-1:
                        split_points.append(segment[-1])
                        continue
                    
                    # 线性插值
                    t = (pos - cum_dists[idx]) / (cum_dists[idx+1] - cum_dists[idx])
                    interp_point = segment[idx] + t * (segment[idx+1] - segment[idx])
                    split_points.append(interp_point)
                
                return np.array(split_points),total_length

            split_segment1, greater_length = split_segment(segment1, pixel_spacing=self.pixel_spacing)  # 大弯均分点 + 弧长
            split_segment2, lesser_length = split_segment(segment2, pixel_spacing=self.pixel_spacing)  # 小弯均分点 + 弧长

            # 7. 合并结果并标记点类型
            combined = np.vstack([split_segment1, split_segment2[1:-1]])  # 避免重复点
            point_types = np.zeros(len(combined), dtype=int)
            point_types[len(split_segment1):] = 1  # 标记第二部分
            
            # 标记关键点
            # for i, point in enumerate(combined):
            #     if np.allclose(point, major_pos, atol=1.0) or np.allclose(point, minor_pos, atol=1.0):
            #         point_types[i] = 2
            # segment1 的长度
            major_nearest = spline_points[major_idx]  # 原始样条上的最近点
            minor_nearest = spline_points[minor_idx]

            major_idx_in_combined = np.argmin(np.linalg.norm(combined - major_nearest, axis=1))
            minor_idx_in_combined = np.argmin(np.linalg.norm(combined - minor_nearest, axis=1))

            # 标记关键点
            point_types[major_idx_in_combined] = 2
            point_types[minor_idx_in_combined] = 2

            # 8. save results
            self.spline_points_per_frame[self.current_index] = {
                'points': combined,
                'point_types': point_types,
                'major_keypoint': major_pos,
                'minor_keypoint': minor_pos,
                'greater_length': greater_length,
                'lesser_length': lesser_length,
                'midpoints': {  
                    'all_midpoints': {'points': [], 'is_keypoint': [], 'endpoints': []},
                    'valid_midpoints': {'points': [], 'is_keypoint': [], 'endpoints': []},
                    'contour_points': []
                }
            }

            # 9. update display
            if not silent:
                self.toggle_split_button.config(state=tk.NORMAL)
                self.status_var.set(f"曲线已分割为两部分，每部分198个点 (共{len(combined)}点)")
                # 显示长度信息
                length_unit = "mm" if self.has_pixel_spacing else "pixels"
                self.status_var.set(f"Curve in this frame has been split - greater curvature length: {greater_length:.2f} {length_unit}, lesser curvature length: {lesser_length:.2f} {length_unit}")
                self.update_roi_display()
            
            return True

        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"分割失败: {str(e)}")
                import traceback
                traceback.print_exc()
            return False

    def calculate_average_lengths(self):
        if not self.spline_points_per_frame:
            messagebox.showwarning("Warning", "No split data available for computation.")
            return

        greater_lengths = []
        lesser_lengths = []

        for frame_data in self.spline_points_per_frame.values():
            major_length = frame_data.get('greater_length', 0.0)
            minor_length = frame_data.get('lesser_length', 0.0)
            if major_length > 0 and minor_length > 0:
                greater_lengths.append(major_length)
                lesser_lengths.append(minor_length)

        if not greater_lengths or not lesser_lengths:
            messagebox.showwarning("Warning", "not found valild length data, please split")
            return

        avg_major = np.mean(greater_lengths)
        avg_minor = np.mean(lesser_lengths)

        ave_major_point_spcaing=avg_major/199
        avg_minor_point_spcaing=avg_minor/199

        spacing_info = ""
        if self.has_pixel_spacing:
            spacing_x, spacing_y = self.pixel_spacing
            spacing_info = f"\n\nCurrent Pixel Spacing:\nX={spacing_x:.4f} mm/pixel, Y={spacing_y:.4f} mm/pixel"


        messagebox.showinfo("average length",
            f"Greater Curvature Avergae Length: {avg_major:.2f} mm\n"
            f"Greater Curvature Avergae Point Spacing: {ave_major_point_spcaing:.2f} mm\n"
            f"Lesser Curvature Avergae Length: {avg_minor:.2f} mm\n"
            f"Lesser Curvature Avergae Point Spacing: {avg_minor_point_spcaing:.2f} mm\n"
            + spacing_info)

    def split_all_frames(self):
        if not self.key_points:
            messagebox.showwarning("Warning", "please mark key points")
            return
        
        frames_to_split = [idx for idx in self.roi_points.keys() 
                        if idx in self.key_points and len(self.key_points[idx]) >= 2]
        
        if not frames_to_split:
            messagebox.showwarning("Warning", "please mark key points")
            return
        
        if not messagebox.askyesno("Confirm", f"Segmentation will be performed on {len(frames_to_split)} frames. Continue?"):
            return
        
        progress_window = tk.Toplevel(self.root)
        progress_label = tk.Label(progress_window, text="processing...")
        progress_label.pack()
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=len(frames_to_split))
        progress_bar.pack()
        progress_window.grab_set()
        
        failed_frames = []
        for i, frame_idx in enumerate(frames_to_split):
            progress_var.set(i+1)
            progress_label.config(text=f"process frame {frame_idx+1}/{len(self.image_files)}")
            progress_window.update()
            
            self.current_index = frame_idx
            self.control_points = self.roi_points[frame_idx].tolist()
            
            if not self.split_at_key_points(silent=True):  
                failed_frames.append(frame_idx + 1)  
        
        progress_window.destroy()
        
        if failed_frames:
            msg = f"完成分割，但{len(failed_frames)}帧不均匀:\n{', '.join(map(str, failed_frames))}"
            messagebox.showwarning("Partially successful, please check", msg)
        else:
            messagebox.showinfo("Success", f"all{len(frames_to_split)} frames successfully spilt!")
        
        self.status_var.set(f"Split completed - success: {len(frames_to_split)-len(failed_frames)} fail: {len(failed_frames)}")

    def validate_key_points(self):
        """Verfy if the key points are valid"""
        if self.current_index not in self.key_points:
            messagebox.showwarning("Warning", "please mark key points")
            return False
        
        key_data = self.key_points[self.current_index]
        if len(key_data) < 2:
            messagebox.showwarning("Warning", " it needs two points")
            return False
        
        if not self.control_points or len(self.control_points) < 3:
            messagebox.showwarning("Warning", "it needs at least 3 control points")
            return False
        
        # Verfy if the key point indices are within the range of control points 
        for ktype, info in key_data.items():
            if info['index'] >= len(self.control_points):
                messagebox.showwarning("Warning", f"{ktype}关键点索引超出范围")
                return False
        
        return True

    def toggle_edit_mode(self):
        if not self.image_files or self.current_index not in self.roi_points:
            messagebox.showwarning("Warning", "No ROI to edit. Please draw an ROI first.")
            return

        if not self.is_editing:
            # Enter editing mode
            self.is_editing = True
            self.is_drawing = False
            self.mode_var.set("Editing ROI")
            self.status_var.set("Press Ctrl+e to enter edit, Right-Click to add points, Right-click to delete points, Press Ctrl+S to finish edit.")

            # Update buttons
            self.draw_button.config(state=tk.DISABLED)
            self.complete_button.config(state=tk.NORMAL)
            self.edit_button.config(state=tk.DISABLED)
        else:
            # Exit editing mode
            self.complete_roi()

    def copy_roi_to_next(self):
        if not self.image_files or self.current_index not in self.roi_points:
            return

        if self.current_index < len(self.image_files) - 1:
            next_index = self.current_index + 1
            self.roi_points[next_index] = np.array(self.control_points)

            # Go to next frame
            self.next_image()
            self.status_var.set("ROI copied to next frame")

    def set_keypoint_type(self, key_type):
        """set type of key point"""
        if not self.control_points:
            messagebox.showwarning("Warning", "please drawing ROI")
            return
        self.current_key_type = key_type
        self.status_var.set(f"Right-click the control point to mark it as {key_type}.")

    def clear_key_points(self):
        """clear key point at current frame"""
        if self.current_index in self.key_points:
            del self.key_points[self.current_index]
        self.update_roi_display()
        self.status_var.set("Key point in curent frame has been cleared")

    def mark_key_point(self, point_idx):
        """Mark the control point with the specified index as a key point"""
        if self.current_index not in self.key_points:
            self.key_points[self.current_index] = {}
        
        point = self.control_points[point_idx]
        self.key_points[self.current_index][self.current_key_type] = {
            'index': point_idx,
            'position': point.copy()
        }
        self.current_key_type = None
        self.status_var.set(f"Marked {self.current_key_type} is key point")
        self.update_roi_display()

    def find_closest_control_point(self, position):
        """find the index of control point closest to the specified position 找到离指定位置最近的控制点索引"""
        if not self.control_points:
            return None
        distances = [np.linalg.norm(np.array(p)-np.array(position)) 
                    for p in self.control_points]
        return np.argmin(distances)

    def update_roi_display(self):
        for marker in self.control_point_markers:
            if marker in self.ax.lines:
                marker.remove()
        if self.spline_line and self.spline_line in self.ax.lines:
            self.spline_line.remove()

        self.control_point_markers = []

        if len(self.control_points) > 0 and self.show_control_points:
            for i, point in enumerate(self.control_points):
                marker, = self.ax.plot(point[0], point[1], 'ro', color='white', markersize=5, picker=5)
                self.control_point_markers.append(marker)

            if self.current_index in self.key_points:
                for ktype, info in self.key_points[self.current_index].items():
                    idx = info['index']
                    if idx < len(self.control_points):
                        point = self.control_points[idx]
                        marker, = self.ax.plot(
                            point[0], point[1], 
                            'o', markersize=self.keypoint_size,
                            color=self.keypoint_colors[ktype],
                            markerfacecolor='none',
                            markeredgewidth=2
                        )
                        self.control_point_markers.append(marker)

        if len(self.control_points) >= 3:
            self.draw_spline()

        if self.current_index in self.spline_points_per_frame and self.show_split_points:
            data = self.spline_points_per_frame[self.current_index]
            split_points = data['points']
            point_types = data['point_types']

            for i, (point, ptype) in enumerate(zip(split_points, point_types)):
                if ptype == 0:  # 第一部分
                    color = 'green'
                    markersize = 3
                elif ptype == 1:  # 第二部分
                    color = 'pink'
                    markersize = 3
                else:  # 关键点
                    color = 'red'
                    markersize = 4
                    
                self.ax.plot(point[0], point[1], 'o', 
                            color=color, 
                            markersize=markersize, 
                            markerfacecolor='none', 
                            markeredgewidth=1)

        if (self.current_index in self.spline_points_per_frame and 
            self.show_midpoints and 
            isinstance(self.spline_points_per_frame[self.current_index].get('midpoints'), dict)):

            mid_data = self.spline_points_per_frame[self.current_index]['midpoints']
            
            all_midpoints = mid_data.get('all_midpoints', {})
            if isinstance(all_midpoints, dict) and len(all_midpoints.get('points', [])) > 0:
                self.ax.scatter(
                    all_midpoints['points'][:, 0], 
                    all_midpoints['points'][:, 1],
                    c='blue', s=5, alpha=0.5, label='All Midpoints'
                )

            valid_midpoints = mid_data.get('valid_midpoints', {})
            if isinstance(valid_midpoints, dict) and len(valid_midpoints.get('points', [])) > 0:
                self.ax.scatter(
                    valid_midpoints['points'][:,0],
                    valid_midpoints['points'][:,1],
                    c='green', s=8, label='Valid Midpoints'
                )
                
                if any([len(mid_data['all_midpoints']) > 0, 
                    len(mid_data['valid_midpoints']) > 0]):
                    self.current_legend = self.ax.legend(loc='upper right')

        self.canvas.draw()

    def draw_spline(self):
        points = np.array(self.control_points)
        points = np.vstack([points, points[0]])  
        
        tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=1)
        u_new = np.linspace(0, 1, self.num_spline_points)
        self.spline_points = np.array(splev(u_new, tck)).T

        if self.spline_line and self.spline_line in self.ax.lines:
            self.spline_line.remove()

        self.spline_line, = self.ax.plot(self.spline_points[:, 0], self.spline_points[:, 1], 'b-', linewidth=2)

    def on_click(self, event):
        if not event.inaxes or not self.image_files:
            return

        # 左键点击
        if event.button == 1:
            if self.is_drawing:
                self.control_points.append([event.xdata, event.ydata])
                self.update_roi_display()

            elif self.is_editing:
                if len(self.control_points) > 0:
                    closest_point_idx = self.find_closest_point(event.xdata, event.ydata)
                    if closest_point_idx is not None:
                        self.selected_point = closest_point_idx

        # right-click
        elif event.button == 3:
            if self.current_key_type and self.is_editing:
                # 标记关键点模式
                closest_point_idx = self.find_closest_point(event.xdata, event.ydata)
                if closest_point_idx is not None:
                    self.mark_key_point(closest_point_idx)
            else:
                if self.is_drawing and len(self.control_points) >= 3:
                    self.complete_roi()
                elif self.is_editing:
                    closest_point_idx = self.find_closest_point(event.xdata, event.ydata)
                    if closest_point_idx is not None:
                        self.remove_point_from_roi(closest_point_idx)
                    else:
                        closest_segment = self.find_closest_segment(event.xdata, event.ydata)
                        if closest_segment is not None:
                            self.insert_control_point_on_line(closest_segment, event.xdata, event.ydata)
                        else:
                            self.add_point_to_roi(event.xdata, event.ydata)

    def insert_control_point_on_line(self, segment_idx, x, y):
        """
        Insert a new control point on the spline line.
        This converts from a spline segment index to the appropriate position
        in the control points list.
        """
        if not self.is_editing or len(self.control_points) < 2:
            return
        
        # Save current state for undo
        self.undo_stack.append({
            "control_points": self.control_points.copy(),
            "roi_points": self.roi_points[self.current_index].copy() if self.current_index in self.roi_points else None
        })
        
        # Find the control points that correspond most closely to this spline segment
        # This requires mapping from spline space to control point space
        spline_length = len(self.spline_points)
        control_length = len(self.control_points)
        
        # Since we have a closed loop, we need to handle the wrap-around case
        if segment_idx == spline_length - 1:
            # This is the segment connecting back to the start
            # Add the new point between the last and first control points
            idx = control_length  # Add at the end (will be connected to the beginning)
        else:
            # Map the spline segment index to a position in the control points list
            # This is an approximation - we insert after the nearest control point
            ratio = segment_idx / spline_length
            idx = max(1, min(control_length, round(ratio * control_length)))
        
        # Insert the new control point
        self.control_points.insert(idx % control_length, [x, y])
        
        # Update the display
        self.update_roi_display()
        self.status_var.set(f"Added new control point on the line")

    def on_release(self, event):
        if self.is_editing and self.selected_point is not None:
            self.selected_point = None
            self.update_roi_display()

    def on_motion(self, event):
        if not event.inaxes or not self.image_files:
            return

        if self.is_editing and self.selected_point is not None:
            # Update the selected control point position
            self.control_points[self.selected_point] = [event.xdata, event.ydata]
            self.update_roi_display()

    def find_closest_point(self, x, y, threshold=10):
        if len(self.control_points) == 0:
            return None

        # Convert data coordinates to display coordinates
        data_coords = np.array(self.control_points)
        disp_coords = np.array([self.ax.transData.transform([cx, cy]) for cx, cy in data_coords])

        # Get mouse position in display coordinates
        mouse_disp = self.ax.transData.transform([x, y])

        # Calculate distances
        distances = np.sqrt(np.sum((disp_coords - mouse_disp)**2, axis=1))

        # Find closest point within threshold
        min_idx = np.argmin(distances)
        if distances[min_idx] <= threshold:
            return min_idx

        return None

    def find_closest_segment(self, x, y, threshold=10):
        """
        Find the closest segment (pair of adjacent points) on the spline line to the clicked position.
        Returns the index of the first point in the segment.
        """
        if len(self.spline_points) < 2:
            return None

        # Convert data coordinates to display coordinates
        spline_coords = np.array(self.spline_points)
        disp_coords = np.array([self.ax.transData.transform([cx, cy]) for cx, cy in spline_coords])

        # Get mouse position in display coordinates
        mouse_disp = self.ax.transData.transform([x, y])

        # Calculate distances to each segment
        min_distance = float('inf')
        closest_segment = None

        for i in range(len(spline_coords)):
            # Get the two points of the segment (handling the wrap-around for the last point)
            p1 = disp_coords[i]
            p2 = disp_coords[(i + 1) % len(disp_coords)]

            # Calculate the distance from the mouse position to the segment
            distance = self.distance_to_segment(mouse_disp, p1, p2)

            if distance < min_distance and distance <= threshold:
                min_distance = distance
                closest_segment = i

        return closest_segment

    def distance_to_segment(self, point, p1, p2):
        """
        Calculate the shortest distance from a point to a line segment defined by p1 and p2.
        """
        # Vector from p1 to p2
        segment_vec = p2 - p1
        # Vector from p1 to the point
        point_vec = point - p1

        # Project point_vec onto segment_vec
        segment_length = np.linalg.norm(segment_vec)
        if segment_length == 0:
            return np.linalg.norm(point_vec)

        t = np.dot(point_vec, segment_vec) / (segment_length ** 2)
        t = max(0, min(1, t))  # Clamp t to the segment

        # Closest point on the segment
        closest_point = p1 + t * segment_vec

        # Distance to the closest point
        return np.linalg.norm(point - closest_point)

    def update_button_states(self):
        # Enable/disable buttons based on current state
        has_roi = self.current_index in self.roi_points and len(self.roi_points[self.current_index]) > 0
        has_next_frame = self.current_index < len(self.image_files) - 1 if self.image_files else False
        has_key_points = (self.current_index in self.key_points and len(self.key_points[self.current_index]) >= 2)
        has_split_points = self.current_index in self.spline_points_per_frame
        has_valid_split_data = (
        self.current_index in self.spline_points_per_frame and 
        'points' in self.spline_points_per_frame[self.current_index] and
        'point_types' in self.spline_points_per_frame[self.current_index]
        )
        
        if has_roi:
            self.clear_button.config(state=tk.NORMAL)
            self.edit_button.config(state=tk.NORMAL)
            self.copy_button.config(state=tk.NORMAL if has_next_frame else tk.DISABLED)
            self.propagate_button.config(state=tk.NORMAL if has_next_frame else tk.DISABLED)
            #self.auto_prop_button.config(state=tk.NORMAL if has_next_frame else tk.DISABLED)
            self.split_button.config(state=tk.NORMAL if has_key_points else tk.DISABLED)
            self.split_all_button.config(state=tk.NORMAL if has_key_points else tk.DISABLED)
            
            can_calculate_midpoints = has_valid_split_data and has_key_points
            self.calc_midpoints_button.config(state=tk.NORMAL if can_calculate_midpoints else tk.DISABLED)
            self.calc_all_midpoints_button.config(state=tk.NORMAL if can_calculate_midpoints else tk.DISABLED)
            
            self.toggle_split_button.config(state=tk.NORMAL if has_valid_split_data else tk.DISABLED)
            self.toggle_control_button.config(state=tk.NORMAL)
            self.toggle_midpoints_button.config(
                state=tk.NORMAL if (has_valid_split_data and 
                                'midpoints' in self.spline_points_per_frame[self.current_index]) 
                else tk.DISABLED
            )
            
        else:
            self.clear_button.config(state=tk.DISABLED)
            self.edit_button.config(state=tk.DISABLED)
            self.copy_button.config(state=tk.DISABLED)
            self.propagate_button.config(state=tk.DISABLED)
            #self.auto_prop_button.config(state=tk.DISABLED)
            self.split_button.config(state=tk.DISABLED)
            self.split_all_button.config(state=tk.DISABLED)
            self.toggle_split_button.config(state=tk.DISABLED)
            self.toggle_control_button.config(state=tk.DISABLED)
            self.calc_midpoints_button.config(state=tk.DISABLED)
            self.calc_all_midpoints_button.config(state=tk.DISABLED)

    def export_images_with_rois(self):
        """export images with contour"""
        if not self.image_files:
            messagebox.showwarning("warning", "No images loaded")
            return
        
        output_dir = filedialog.askdirectory(title="select output folder")
        if not output_dir:
            return
        
        # 创建进度窗口
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Exporting images")
        progress_label = tk.Label(progress_window, text="exporting images ...")
        progress_label.pack(pady=10)
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, 
                                    maximum=len(self.image_files))
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_window.grab_set()
        
        try:
            for i, img_path in enumerate(self.image_files):
                progress_var.set(i+1)
                progress_label.config(text=f"Exporting {i+1}/{len(self.image_files)}")
                progress_window.update()
                
                # 加载原始图像
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img)
                
                # 创建与原始图像相同尺寸的图形
                dpi = 100
                height, width = img_array.shape[:2]
                figsize = width / float(dpi), height / float(dpi)
                
                fig = plt.figure(figsize=figsize, dpi=dpi)
                ax = fig.add_axes([0, 0, 1, 1])  # 使坐标轴覆盖整个图形
                ax.axis('off')
                
                # 显示图像
                if len(img_array.shape) == 2 or img_array.shape[2] == 1:
                    ax.imshow(img_array, cmap='gray')
                else:
                    ax.imshow(img_array)
                
                # 如果该帧有ROI数据，则绘制ROI
                if i in self.roi_points and len(self.roi_points[i]) > 0:
                    # 绘制控制点
                    control_points = self.roi_points[i]
                    #ax.plot(control_points[:, 0], control_points[:, 1], 'ro', markersize=3)
                    
                    # 绘制样条曲线
                    points = np.vstack([control_points, control_points[0]])  # 闭合曲线
                    tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=1)
                    u_new = np.linspace(0, 1, 1000)
                    spline_points = np.array(splev(u_new, tck)).T
                    ax.plot(spline_points[:, 0], spline_points[:, 1], 'lime', linewidth=1)
                    
                    # 如果有关键点，绘制关键点
                    if i in self.key_points:
                        for ktype, info in self.key_points[i].items():
                            idx = info['index']
                            if idx < len(control_points):
                                point = control_points[idx]
                                color = 'gold' if ktype == 'major' else 'cyan'
                                ax.plot(point[0], point[1], 'o', markersize=3,
                                    color=color, markerfacecolor='none',
                                    markeredgewidth=2)
                    
                    if hasattr(self, 'centerline_data') and i in self.centerline_data['frame_indices']:
                        centerline = self.centerline_data['centerline_points']
                        
                        # 确保中心线在图像范围内
                        #centerline = np.clip(centerline, [0, 0], [width-1, height-1])
                        
                        # 绘制中心线（红色，较粗）
                        ax.plot(centerline[:, 0], centerline[:, 1], 'r-', linewidth=2)
                        
                        # 标记中心线起点和终点
                        # ax.plot(centerline[0, 0], centerline[0, 1], 'go', markersize=6, label='起点')
                        # ax.plot(centerline[-1, 0], centerline[-1, 1], 'yo', markersize=6, label='终点')
                
                output_path = os.path.join(output_dir, f"frame_{i:04d}_with_roi.png")
                fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
                plt.close(fig)
            
            messagebox.showinfo("Success", f"Exported {len(self.image_files)} images to:\n{output_dir}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
        
        finally:
            progress_window.destroy()

    def save_rois(self):
        if not self.image_files or not self.roi_points:
            messagebox.showwarning("Warning", "No ROIs to save")
            return

        try:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".npy",
                filetypes=[("NumPy Files", "*.npy"), ("All Files", "*.*")],
                title="Save ROIs"
            )

            if save_path:
                # Convert dictionary to serializable format
                roi_data = {}
                for idx, points in self.roi_points.items():
                    roi_data[idx] = np.array(points)

                np.save(save_path, roi_data)
                messagebox.showinfo("Success", f"ROIs saved to {save_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Error saving ROIs: {str(e)}")
    
    def save_splinerois(self):
        """Save wall points and their type"""
        if not self.spline_points_per_frame:
            messagebox.showwarning("Warning", "No spline points to save")
            return

        try:

            # verify the order consistency of all frames 验证所有帧的顺序一致性
            first_frame_order = None
            inconsistent_frames = []
            
            for idx, data in self.spline_points_per_frame.items():
                # get position of key point 获取关键点位置
                major_pos = data['major_keypoint']
                minor_pos = data['minor_keypoint']
                
                # check the order
                current_order = (major_pos[0] > minor_pos[0])  # True表示大弯在右
                
                if first_frame_order is None:
                    first_frame_order = current_order
                elif current_order != first_frame_order:
                    inconsistent_frames.append(idx)
            
            if inconsistent_frames:
                messagebox.showwarning("Warning", 
                    f"发现{len(inconsistent_frames)}帧顺序不一致:\n"
                    f"帧号: {inconsistent_frames}\n"
                    "建议重新处理这些帧以确保一致性")
                return
        
            save_path = filedialog.asksaveasfilename(
                defaultextension=".npz",
                filetypes=[("NumPy ZIP Archive", "*.npz"), ("All Files", "*.*")],
                title="Save All Spline Points with Types"
            )

            if save_path:
                frame_indices = []
                all_points = []
                all_types = []
                major_points = []
                minor_points = []
                
                for idx, data in sorted(self.spline_points_per_frame.items()):
                    frame_indices.append(idx)
                    all_points.append(data['points'])
                    all_types.append(data['point_types'])
                    major_points.append(data['major_keypoint'])
                    minor_points.append(data['minor_keypoint'])
                
                # 使用npz格式保存多个数组
                np.savez(
                    save_path,
                    frame_indices=np.array(frame_indices),
                    points=np.array(all_points),
                    point_types=np.array(all_types),
                    major_points=np.array(major_points),
                    minor_points=np.array(minor_points)
                )
                
                messagebox.showinfo("Success", f"splineROIs saved to {save_path}\n"
                                            f"including {len(frame_indices)} frames\n"
                                            f"point type: 0=greater curvature, 1=lesser curvature, 2=key point")

        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}")

    def calculate_valid_midpoints(self, frame_idx):
        """
        计算有效中点（基于样条曲线边界验证）
        返回结构：
        {
            'all_midpoints': 所有中点坐标,
            'valid_midpoints': 有效中点坐标,
            'keypoint1': 大弯关键点,
            'keypoint2': 小弯关键点
        }
        """
        if frame_idx not in self.spline_points_per_frame or frame_idx not in self.key_points:
            return None
        
        data = self.spline_points_per_frame[frame_idx]
        points = data['points']
        point_types = data['point_types']
        
        # 获取关键点(这里使用手动选取的大弯点或者小弯点，而不是split at points中的类型2)
        key_data = self.key_points[frame_idx]
        if 'major' not in key_data or 'minor' not in key_data:
            return None       
        keypoint1 = key_data['major']['position']  # 大弯点
        keypoint2 = key_data['minor']['position']  # 小弯点

        # # 获取关键点(这里使用split at points中的类型2)       
        # key_points = points[point_types == 2]
        # if len(key_points) < 2:
        #     return None       
        # # 确定大弯点和小弯点（假设第一个是major，第二个是minor）
        # keypoint1 = key_points[0]  # 大弯点
        # keypoint2 = key_points[1]  # 小弯点
        
        # 分割小弯和大弯点
        minor_points = points[point_types == 0]  # 小弯点
        major_points = points[point_types == 1]  # 大弯点
        
        # 确保点数相同且成对
        n_points = min(len(minor_points), len(major_points))
        minor_points = minor_points[:n_points]
        major_points = major_points[:n_points]
        
        # 获取当前帧的样条曲线点（1000个插值点）
        if frame_idx not in self.roi_points or len(self.roi_points[frame_idx]) < 3:
            return None
        
        # 生成样条曲线边界路径（自动闭合）
        roi_points = np.array(self.roi_points[frame_idx])
        tck, u = splprep([roi_points[:,0], roi_points[:,1]], s=0, per=1)
        u_new = np.linspace(0, 1, 1000)
        spline_points = np.array(splev(u_new, tck)).T
        roi_path = Path(np.vstack([spline_points, spline_points[0]]))  # 闭合路径
        
        # 初始化结果结构
        results = {
            'all_midpoints': {
                'points': [],          # 中点坐标
                'is_keypoint': [],     # 是否关键点
                'endpoints': []        # 对应的端点对
            },
            'valid_midpoints': {
                'points': [],
                'is_keypoint': [],
                'endpoints': []
            },
            'contour_points': spline_points
        }
        
        # 首先添加关键点（不验证直接包含）
        results['all_midpoints']['points'].extend([keypoint1, keypoint2])
        results['all_midpoints']['is_keypoint'].extend([True, True])
        results['all_midpoints']['endpoints'].extend([(None, None), (None, None)])  # 关键点无端点对
        
        results['valid_midpoints']['points'].extend([keypoint1, keypoint2])
        results['valid_midpoints']['is_keypoint'].extend([True, True])
        results['valid_midpoints']['endpoints'].extend([(None, None), (None, None)])
        
        # 计算其他中点并验证
        for i in range(n_points):
            minor_pt = minor_points[i]
            major_pt = major_points[n_points - 1 - i]
            midpoint = (minor_pt + major_pt) / 2
            
            # 验证逻辑
            distance = np.linalg.norm(minor_pt - major_pt)
            radius = distance * 0.9 / 2
            angles = np.deg2rad(np.arange(0, 360, 15))
            circle_pts = np.column_stack([
                midpoint[0] + radius * np.cos(angles),
                midpoint[1] + radius * np.sin(angles)
            ])
            
            # 添加到所有中点
            results['all_midpoints']['points'].append(midpoint)
            results['all_midpoints']['is_keypoint'].append(False)
            results['all_midpoints']['endpoints'].append((minor_pt, major_pt))
            
            # 验证通过则添加到有效中点
            if np.all(roi_path.contains_points(circle_pts)):
                results['valid_midpoints']['points'].append(midpoint)
                results['valid_midpoints']['is_keypoint'].append(False)
                results['valid_midpoints']['endpoints'].append((minor_pt, major_pt))
        
        # 转换为numpy数组
        for category in ['all_midpoints', 'valid_midpoints']:
            results[category]['points'] = np.array(results[category]['points'])
            results[category]['is_keypoint'] = np.array(results[category]['is_keypoint'])
            results[category]['endpoints'] = np.array(
                results[category]['endpoints'],
                dtype=[('endpoint1', 'f8', 2), ('endpoint2', 'f8', 2)]
            )
        
        results['all_midpoints_array'] = results['all_midpoints']['points']
        results['valid_midpoints_array'] = results['valid_midpoints']['points']

        self.spline_points_per_frame[frame_idx]['midpoints'] = results
        return results
    
    def calculate_current_midpoints(self):
        """calculate valid midpoints for current frame 计算当前帧的有效中点"""
        if self.current_index not in self.spline_points_per_frame:
            messagebox.showwarning("Warning", "Please split the current frame first")
            return
        
        result = self.calculate_valid_midpoints(self.current_index)
        if result:
            self.update_roi_display()
            valid_count = len(result['valid_midpoints']['points'])
            self.status_var.set(f"The number of valid midpoints at current frame: {valid_count}")
            messagebox.showinfo("Success", 
                f"- all midpoints: {len(result['all_midpoints']['points']) }\n"
                f"- valid midpoints: {valid_count}\n")
        else:
            messagebox.showerror("Error", "calculate failed")

    def calculate_all_midpoints(self):
        """calculate valid midpoints for all frames"""
        if not self.spline_points_per_frame:
            messagebox.showwarning("Warning", "Please split frame first")
            return
        
        if not messagebox.askyesno("Confirm", f"Compute midpoints for {len(self.spline_points_per_frame)} frames. Continue?"):
            return
        
        progress_window = tk.Toplevel(self.root)
        progress_label = tk.Label(progress_window, text="processing...")
        progress_label.pack()
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, 
                                    maximum=len(self.spline_points_per_frame))
        progress_bar.pack()
        progress_window.grab_set()
        
        failed_frames = []
        valid_counts = []
        
        for i, frame_idx in enumerate(sorted(self.spline_points_per_frame.keys())):
            progress_var.set(i+1)
            progress_label.config(text=f"Process frame {frame_idx+1}/{len(self.image_files)}")
            progress_window.update()
            
            try:
                result = self.calculate_valid_midpoints(frame_idx)
                if result:
                    valid_counts.append(len(result['valid_midpoints']['points']))
                else:
                    failed_frames.append(frame_idx + 1)
            except:
                failed_frames.append(frame_idx + 1)

        progress_window.destroy()
        
        if valid_counts:
            avg_valid = np.mean(valid_counts)
            summary = (
                f"Complete:\n"
                f"- frames: {len(self.spline_points_per_frame)}\n"
                f"- success: {len(self.spline_points_per_frame)-len(failed_frames)}\n"
                f"- fail: {len(failed_frames)}\n"
                f"- average valid midpoints: {avg_valid:.1f}"
            )
            
            if failed_frames:
                summary += f"\n Failed frame: {', '.join(map(str, failed_frames))}"
            
            messagebox.showinfo("Results", summary)
            self.status_var.set(f"Complete - average valid midpoints: {avg_valid:.1f}")
        else:
            messagebox.showerror("Error", "Calculate failed for all frames")

    def fit_polynomial_to_midpoints(self):
        """Fit the midpoints of all time steps using a quintic polynomial"""
        try:
            # 1. collect all midpoints
            all_midpoints = []
            frame_indices = []
            
            for frame_idx in sorted(self.spline_points_per_frame.keys()):
                if 'midpoints' not in self.spline_points_per_frame[frame_idx]:
                    continue
                    
                mid_data = self.spline_points_per_frame[frame_idx]['midpoints']
                
                # use all midpoints not the valid points
                if 'all_midpoints' in mid_data and len(mid_data['all_midpoints']['points']) > 0:
                    all_midpoints.append(mid_data['all_midpoints']['points'])
                    frame_indices.append(frame_idx)
            
            if len(all_midpoints) < 3:
                messagebox.showwarning("Warning", "Too few frames to fit the centreline. Please check")
                return
            
            # 2. 合并所有帧的中点
            combined_midpoints = np.vstack(all_midpoints)
            
            # 3. 定义五阶多项式函数
            def polynomial_5th_order(x, a, b, c, d, e, f):
                return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f
            
            # 4. 提取x和y坐标
            x = combined_midpoints[:, 0]
            y = combined_midpoints[:, 1]
            
            # 5. 拟合五阶多项式
            popt, pcov = curve_fit(polynomial_5th_order, x, y)
            
            # 6. 生成200个等间距点的中心线
            x_line = np.linspace(x.min()*1.03, x.max()*0.99, 200)
            y_line = polynomial_5th_order(x_line, *popt)
            centerline_points = np.column_stack((x_line, y_line))
            
            # 7. 保存结果
            self.centerline_data = {
                'fitted_coefficients': popt,
                'covariance_matrix': pcov,
                'centerline_points': centerline_points,
                'frame_indices': frame_indices,
                'original_midpoints': all_midpoints
            }
            
            # 确保 centerline_plots 已初始化
            if not hasattr(self, 'centerline_plots'):
                self.centerline_plots = []

            # 8. 可视化结果
            self.visualize_centerline()
            
            # 9. 显示成功消息
            messagebox.showinfo("Success", 
                f"- frames: {len(all_midpoints)}\n"
                f"- all midpoints: {len(combined_midpoints)}\n"
                f"- points of centreline: 200")
        
        except Exception as e:
            messagebox.showerror("Error", f"Centreline fitting failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def visualize_centerline(self):
        if not hasattr(self, 'centerline_data') or not self.show_centerline:
            return
        
        for artist in self.centerline_plots:
            try:
                if artist in self.ax.lines or artist in self.ax.collections:
                    artist.remove()
            except:
                pass  #
        self.centerline_plots = []  
        
        centerline_points = self.centerline_data['centerline_points']
        original_midpoints = self.centerline_data['original_midpoints']
        
        line, = self.ax.plot(
            centerline_points[:, 0], centerline_points[:, 1],
            'r-', markersize=8, linewidth=3, label='Fitted Centerline'
        )
        self.centerline_plots.append(line)
        
        if not self.ax.get_legend():
            self.ax.legend(loc='upper right')
        # if not any(isinstance(child, plt.legend) for child in self.ax.get_children()):
        #     self.ax.legend(loc='upper right')
        
        self.canvas.draw()
        self.status_var.set(f"Display the fitted centerline ({len(original_midpoints)} frames)")

    def save_centerline_data(self):
        if not hasattr(self, 'centerline_data'):
            messagebox.showwarning("Warning", "no centerline data")
            return
        
        try:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".npz",
                filetypes=[("NumPy ZIP Archive", "*.npz"), ("All Files", "*.*")],
                title="Save Centerline Data"
            )
            
            if save_path:
                # 准备数据 - 使用正确的数据结构
                np.savez(
                    save_path,
                    # fitted_coefficients=self.centerline_data['fitted_coefficients'],
                    # covariance_matrix=self.centerline_data['covariance_matrix'],
                    centerline_points=self.centerline_data['centerline_points']
                    # frame_indices=self.centerline_data['frame_indices'],
                    # original_midpoints=self.centerline_data['original_midpoints']
                )
                
                messagebox.showinfo("Success", f"Centerline data has been saved: {save_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def calculate_optical_flow(self, prev_frame, curr_frame):
        """
        Calculate optical flow between two consecutive frames using Farneback algorithm.
        """
        try:
            # Ensure images are in grayscale
            if len(prev_frame.shape) == 3:
                prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            if len(curr_frame.shape) == 3:
                curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0
            )

            return flow
        except Exception as e:
            print(f"Error calculating optical flow: {e}")
            return None
    
    def propagate_roi_with_flow(self):
        if self.current_index >= len(self.image_files) - 1:
            messagebox.showwarning("Warning", "no available next frame")
            return

        if self.current_index not in self.roi_points:
            messagebox.showwarning("Warning", "no propgate ROI")
            return

        try:
            current_frame = Image.open(self.image_files[self.current_index]).convert('L')
            current_frame = np.array(current_frame)
            next_frame = Image.open(self.image_files[self.current_index + 1]).convert('L')
            next_frame = np.array(next_frame)

            flow = self.calculate_optical_flow(current_frame, next_frame)

            propagated_points = []
            for point in self.roi_points[self.current_index]:
                x, y = point
                dx = flow[int(y), int(x), 0]
                dy = flow[int(y), int(x), 1]
                propagated_points.append([x + dx, y + dy])

            self.roi_points[self.current_index + 1] = np.array(propagated_points)

            if self.current_index in self.key_points:
                prev_keypoints = self.key_points[self.current_index]
                new_keypoints = {}
                
                for ktype, info in prev_keypoints.items():
                    old_idx = info['index']
                    if old_idx >= len(self.control_points):
                        continue
                    
                    x, y = self.control_points[old_idx]
                    dx = flow[int(y), int(x), 0]
                    dy = flow[int(y), int(x), 1]
                    
                    new_pos = [x+dx, y+dy]
                    new_idx = self.find_closest_control_point(new_pos)
                    
                    new_keypoints[ktype] = {
                        'index': new_idx,
                        'position': propagated_points[new_idx]
                    }
                
                self.key_points[self.current_index + 1] = new_keypoints

            self.current_index += 1
            self.display_current_image()
            self.status_var.set("ROI and key points have propagated to next frame")

        except Exception as e:
            messagebox.showerror("Error", f"Propagate failed: {str(e)}")

    def auto_propagate_all_frames(self):
        """
        Automatically propagate the ROI to all subsequent frames using optical flow.
        """
        if self.current_index >= len(self.image_files) - 1:
            messagebox.showwarning("Warning", "No frames available for propagation")
            return

        if self.current_index not in self.roi_points:
            messagebox.showwarning("Warning", "No ROI to propagate")
            return

        try:
            # Disable buttons during propagation
            self.propagate_button.config(state=tk.DISABLED)
            #self.auto_prop_button.config(state=tk.DISABLED)
            self.root.update()

            # Initialize progress bar
            self.progress_var.set(0)
            self.progress_bar["maximum"] = len(self.image_files) - self.current_index - 1

            # Propagate ROI to all subsequent frames
            for i in range(self.current_index, len(self.image_files) - 1):
                # Load current and next frame
                current_frame = Image.open(self.image_files[i]).convert('L')
                current_frame = np.array(current_frame)
                next_frame = Image.open(self.image_files[i + 1]).convert('L')
                next_frame = np.array(next_frame)

                # Calculate optical flow
                flow = self.calculate_optical_flow(current_frame, next_frame)

                # Propagate ROI points
                propagated_points = []
                for point in self.roi_points[i]:
                    x, y = point
                    dx = flow[int(y), int(x), 0]
                    dy = flow[int(y), int(x), 1]
                    propagated_points.append([x + dx, y + dy])

                # Save propagated points to the next frame
                self.roi_points[i + 1] = np.array(propagated_points)

                # Update progress bar
                self.progress_var.set(i - self.current_index + 1)
                self.root.update()

            # Move to the last frame
            self.current_index = len(self.image_files) - 1
            self.display_current_image()
            self.status_var.set("ROI propagated to all frames using optical flow")

        except Exception as e:
            messagebox.showerror("Error", f"Error during auto propagation: {str(e)}")
        finally:
            # Re-enable buttons
            self.propagate_button.config(state=tk.NORMAL)
            #self.auto_prop_button.config(state=tk.NORMAL)

    def display_optical_flow(self):
        """
        Display optical flow visualization on the image.
        """
        if self.current_index >= len(self.image_files) - 1:
            return  # No next frame available

        try:
            # Load current frame and next frame
            current_frame = Image.open(self.image_files[self.current_index]).convert('L')
            current_frame = np.array(current_frame)
            next_frame = Image.open(self.image_files[self.current_index + 1]).convert('L')
            next_frame = np.array(next_frame)

            # Calculate optical flow
            flow = self.calculate_optical_flow(current_frame, next_frame)
            if flow is None:
                print("Optical flow calculation failed.")
                return

            # Draw optical flow arrows
            step = 10  # Increased step size for better visibility
            h, w = flow.shape[:2]
            y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
            fx, fy = flow[y, x].T

            # Calculate optical flow magnitude (for color mapping)
            magnitude = np.sqrt(fx**2 + fy**2)
            print(f"Optical flow magnitude range: {np.min(magnitude)} to {np.max(magnitude)}")

            # Clear previous optical flow arrows
            for quiver in self.flow_quiver:
                try:
                    quiver.remove()
                except:
                    pass  # Handle case where quiver is already removed
            self.flow_quiver = []

            # Store current image position and size
            bbox_before = self.ax.get_position()
            
            # Turn off autoscaling before drawing quiver
            self.ax.autoscale(False)
            
            # Draw optical flow arrows with improved visibility
            quiver = self.ax.quiver(
                x, y, fx, fy, magnitude, angles='xy',  scale_units='xy', scale=0.5,  # Decreased scale to make arrows more visible
                cmap='inferno', 
                width=0.005,  # Slightly wider arrows
                minshaft=2,   # Minimum length of arrow shaft
                minlength=0.2  # Minimum arrow length
            )
            
            if quiver is not None:
                self.flow_quiver.append(quiver)
            else:
                print("Failed to create quiver object.")

            # Add colorbar - use a dedicated axis for colorbar
            if self.flow_colorbar:
                try:
                    self.flow_colorbar.remove()
                except:
                    pass  # Handle case where colorbar is already removed
                self.flow_colorbar = None
                
            # If colorbar axis exists, use it, otherwise create one
            if not hasattr(self, 'cax') or self.cax is None:
                # Create a new axis for the colorbar
                # Position it to the right of the main plot without affecting main plot size
                cax_position = [0.92, 0.15, 0.02, 0.7]  # [left, bottom, width, height]
                self.cax = self.fig.add_axes(cax_position)
            
            # Create colorbar using the dedicated axis
            self.flow_colorbar = self.fig.colorbar(quiver, cax=self.cax, label='Optical Flow Magnitude')
            
            # Restore main axis position if it changed
            bbox_after = self.ax.get_position()
            if bbox_before != bbox_after:
                self.ax.set_position(bbox_before)

            # Make sure limits stay consistent with the image dimensions
            if hasattr(self, 'xlim') and hasattr(self, 'ylim'):
                self.ax.set_xlim(self.xlim)
                self.ax.set_ylim(self.ylim)
            else:  
                self.ax.set_xlim(0, w)
                self.ax.set_ylim(h, 0)  # Inverted y-axis for image coordinates
                
            self.ax.autoscale(False)  # 关闭自动缩放
            
            self.canvas.draw()

        except Exception as e:
            print(f"Error displaying optical flow: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging

    def toggle_optical_flow(self):
        self.show_flow = not self.show_flow
        if self.show_flow:
            self.display_optical_flow()
            self.status_var.set("Optical flow displayed")
        else:
            for quiver in self.flow_quiver:
                try:
                    quiver.remove()
                except:
                    pass
            self.flow_quiver = []

            # 清除 colorbar，但保留其轴
            if self.flow_colorbar:
                try:
                    self.flow_colorbar.remove()
                except:
                    pass
                self.flow_colorbar = None
                
            # 如果有 cax，清除其内容但保留轴对象
            if hasattr(self, 'cax') and self.cax is not None:
                self.cax.clear()

            # 恢复主坐标轴位置和范围
            # 获取当前图像尺寸
            if hasattr(self, 'current_image') and self.current_image is not None:
                h, w = self.current_image.shape[:2] if len(self.current_image.shape) > 2 else (self.current_image.shape[0], self.current_image.shape[1])
            
                # 使用固定的标准位置
                if not hasattr(self, 'standard_position'):
                    # 可以从初始化或其他地方获取标准位置
                    self.standard_position = [0.125, 0.11, 0.775, 0.77]  # 标准 matplotlib 位置 [left, bottom, width, height]
                
                # 恢复到标准位置
                self.ax.set_position(self.standard_position)
                
                # 恢复轴范围
                if hasattr(self, 'xlim') and hasattr(self, 'ylim'):
                    self.ax.set_xlim(self.xlim)
                    self.ax.set_ylim(self.ylim)
                else:
                    self.ax.set_xlim(0, w)
                    self.ax.set_ylim(h, 0)  # 注意 y 轴是倒置的
                
                self.ax.autoscale(False)  # 确保关闭自动缩放

            self.canvas.draw()
            self.status_var.set("Optical flow hidden")

    def toggle_split_points(self):
        self.show_split_points = not self.show_split_points

        if self.show_split_points:
            self.toggle_split_button.config(text="Hide Split Points")
            self.status_var.set("Split points are now visible")
        else:
            self.toggle_split_button.config(text="Show Split Points")
            self.status_var.set("Split points are now hidden")

        self.update_roi_display()

        def undo(self):
            """
            Undo the last action.
            """
            if len(self.undo_stack) > 0:
                # Restore the previous state
                prev_state = self.undo_stack.pop()
                self.control_points = prev_state["control_points"]
                self.roi_points[self.current_index] = prev_state["roi_points"]
                self.update_roi_display()
                self.status_var.set("Undo: Last action reverted")
            else:
                self.status_var.set("Undo: No actions to undo")

    def toggle_control_points(self):
        self.show_control_points = not self.show_control_points
        
        if self.show_control_points:
            self.toggle_control_button.config(text="Hide Control Points")
            self.status_var.set("Control points are now visible")
        else:
            self.toggle_control_button.config(text="Show Control Points")
            self.status_var.set("Control points are now hidden")
        
        self.update_roi_display()

    def toggle_midpoints(self):
        self.show_midpoints = not self.show_midpoints
        
        if self.show_midpoints:
            self.toggle_midpoints_button.config(text="Hide Midpoints")
            self.status_var.set("Midpoints are now visible")
        else:
            self.toggle_midpoints_button.config(text="Show Midpoints")
            self.status_var.set("Midpoints are now hidden")
        
        self.update_roi_display()

    def toggle_centerline(self):
        self.show_centerline = not self.show_centerline
        
        if hasattr(self, 'centerline_data'):
            if self.show_centerline:
                self.visualize_centerline()
            else:
                for artist in self.centerline_plots:
                    try:
                        if artist in self.ax.lines or artist in self.ax.collections:
                            artist.remove()
                    except:
                        pass
                self.centerline_plots = []
                self.canvas.draw()
        
        self.status_var.set(f"Centerline {'visible' if self.show_centerline else 'hidden'}")

    def add_point_to_roi(self, x, y):
        """
        Add a point to the ROI.
        """
        if self.is_editing:
            # Save current state for undo
            self.undo_stack.append({
                "control_points": self.control_points.copy(),
                "roi_points": self.roi_points[self.current_index].copy()
            })

            # Add the new point at the clicked position
            self.control_points.append([x, y])
            self.update_roi_display()

    def remove_point_from_roi(self, index):
        """
        Remove a point from the ROI.
        """
        if self.is_editing and 0 <= index < len(self.control_points):
            # Save current state for undo
            self.undo_stack.append({
                "control_points": self.control_points.copy(),
                "roi_points": self.roi_points[self.current_index].copy()
            })

            # Remove the point
            self.control_points.pop(index)
            self.update_roi_display()


# Main application loop
if __name__ == "__main__":
    root = tk.Tk()
    app = MRISegmentationApp(root)
    root.mainloop()