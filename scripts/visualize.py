#!/usr/bin/env python3
"""Visualize segmentation with bounding boxes and labels on the original image."""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from component import ComponentSegmenter, CLASS_NAMES


def get_distinct_colors(n):
    """Generate n visually distinct colors."""
    colors = [
        '#FF0000',  # Red
        '#00FF00',  # Green
        '#0000FF',  # Blue
        '#FFFF00',  # Yellow
        '#FF00FF',  # Magenta
        '#00FFFF',  # Cyan
        '#FF8000',  # Orange
        '#8000FF',  # Purple
        '#FF0080',  # Pink
        '#80FF00',  # Lime
        '#0080FF',  # Sky Blue
        '#FF8080',  # Light Red
    ]
    return colors[:n]


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def visualize_detections(image_path, result, mask, output_path):
    """Create visualization with bounding boxes, labels, and per-instance masks overlaid."""
    # Load original image
    img = Image.open(image_path)
    img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to 0-1
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Start with the original image
    display_img = img_np.copy()
    
    # Get colors for each detection
    colors = get_distinct_colors(len(result.class_ids))
    
    # If we have per-instance masks, overlay them
    if hasattr(result, 'masks') and result.masks is not None:
        # Draw each instance mask with its own color
        for i in range(result.masks.shape[2]):
            instance_mask = result.masks[:, :, i]
            color_rgb = hex_to_rgb(colors[i])
            
            # Create a colored mask for this instance
            colored_mask = np.zeros_like(display_img)
            for c in range(3):
                colored_mask[:, :, c] = color_rgb[c]
            
            # Blend this instance mask with the display image (40% mask, 60% original)
            alpha = instance_mask[:, :, np.newaxis] * 0.4
            display_img = display_img * (1 - alpha) + colored_mask * alpha
    
    # Show the blended result
    ax.imshow(display_img)
    
    # Draw bounding boxes and labels on top
    for i, (roi, class_id, score) in enumerate(zip(result.rois, result.class_ids, result.scores)):
        # ROI format: [y1, x1, y2, x2]
        y1, x1, y2, x2 = roi
        width = x2 - x1
        height = y2 - y1
        
        # Draw bounding box
        rect = Rectangle((x1, y1), width, height,
                        linewidth=3, edgecolor=colors[i],
                        facecolor='none', linestyle='-')
        ax.add_patch(rect)
        
        # Create label
        class_name = CLASS_NAMES[class_id]
        label = f'{class_name} {score:.1%}'
        
        # Draw label background
        ax.text(x1, y1 - 8, label,
               bbox=dict(facecolor=colors[i], alpha=0.95, edgecolor='white', boxstyle='round,pad=0.5'),
               fontsize=11, color='white', weight='bold',
               verticalalignment='bottom')
        
        # Print mask info
        if hasattr(result, 'masks') and result.masks is not None and i < result.masks.shape[2]:
            instance_mask = result.masks[:, :, i]
            mask_pixels = np.sum(instance_mask)
            print(f"  {i+1}. {class_name}: bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                  f"size={width:.0f}x{height:.0f} mask_pixels={mask_pixels:,} conf={score:.2%}")
        else:
            print(f"  {i+1}. {class_name}: bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                  f"size={width:.0f}x{height:.0f} conf={score:.2%}")
    
    ax.axis('off')
    ax.set_title(f'Detected {len(result.class_ids)} Components with Instance Masks', 
                fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"\n✓ Visualization saved to {output_path}")


def main():
    if len(sys.argv) < 2:
        image_path = 'assets/input.png'
        output_path = 'assets/output.png'
    else:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'assets/output.png'
    
    print(f"Processing: {image_path}")
    
    # Run segmentation
    model_path = 'models/model.h5' if os.path.exists('models/model.h5') else 'model.h5'
    segmenter = ComponentSegmenter(model_path)
    result, mask = segmenter.segment_image_path(image_path)
    
    print(f"\nDetected {len(result.class_ids)} components:")
    
    # Create visualization
    visualize_detections(image_path, result, mask, output_path)
    
    # Print mask statistics
    print(f"\nMask statistics:")
    unique_vals = np.unique(mask)
    print(f"  Unique class IDs in mask: {unique_vals}")
    for val in unique_vals:
        pixel_count = np.sum(mask == val)
        percentage = (pixel_count / mask.size) * 100
        class_name = CLASS_NAMES[val] if val < len(CLASS_NAMES) else "Unknown"
        print(f"    Class {val} ({class_name}): {pixel_count:,} pixels ({percentage:.2f}%)")
    
    print(f"\n💡 Why only {len(unique_vals)} classes in mask but {len(result.class_ids)} detections?")
    print(f"   Some detected bounding boxes may have very small or no pixels in the final mask")
    print(f"   due to overlap, occlusion, or post-processing thresholds.")


if __name__ == '__main__':
    main()
