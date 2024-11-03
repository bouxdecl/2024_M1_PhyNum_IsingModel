import numpy as np
import scipy
import scipy.signal as signal
import scipy.constants
import scipy.optimize
import scipy.ndimage

import matplotlib.pyplot as plt

Tc = 2.269

def create_kernel(upper_left_quarter):
    n_quarter = len(upper_left_quarter)
    N = 2 * n_quarter - 1
    matrix = np.array(upper_left_quarter)
    top_half = np.hstack([matrix, matrix[:, :-1][:, ::-1]])
    return np.vstack([top_half, top_half[:-1, :][::-1, :]])


def neighbour_manhattan(d):
    kernel = create_kernel(np.fliplr(np.identity(d+1))) #matrice avec des 1 à la distance de Manhattan d du centre 
    return kernel/np.sum(kernel) #on normalise pour avoir toujours le même nombre de voisons distant de d 


def create_circular_kernel(d,eps=0.5): #eps = 0.5 de base mais semble mieux avec 0.1

    center = d 
    y, x = np.ogrid[-center:center+1, -center:center+1]

    dist_from_center = np.sqrt(x**2 + y**2)

    kernel = (np.abs(dist_from_center - d) < eps).astype(float)
    
    return kernel/np.sum(kernel)


def x_correlation_kernel(d):
    kernel = np.zeros((1, 2*d + 1))
    kernel[0, 2*d] = 1  
    kernel[0, 0] = 1  

    return kernel/np.sum(kernel)


def y_correlation_kernel(d):
    kernel = np.zeros((2*d + 1,1))
    kernel[2*d,0] = 1  
    kernel[0, 0] = 1  
 
    return kernel/np.sum(kernel)


kernels = {
    'first_neig': create_kernel([[0, 1], 
                                 [1, 0]]),

    'second_neig': create_kernel([[0, 0, 1], 
                                  [0, 1, 2], 
                                  [1, 2, 0]]),

    'repulsive_corners': create_kernel([[-3, 0, 1], 
                                        [0 , 1, 2], 
                                        [1 , 2, 0]]),
    'triangular': np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
}


def padd_opposite_bound(A):
    N = len(A)
    A_padded = np.pad(A, 1)
    A_padded[0]  = 1
    A_padded[-1] = -1
    A_padded[:N//2, 0]  = 1
    A_padded[:N//2, -1] = 1

    A_padded[N//2:, 0]  = -1
    A_padded[N//2:, -1] = -1

    return A_padded


def remove_non_border_components(binary_image):
    """
    Removes connected components from a binary image that do not touch the border.
    
    Parameters:
        binary_image (numpy.ndarray): The input binary image (2D array).
        
    Returns:
        numpy.ndarray: A binary image with only the components that touch the border.
    """
    # Label the connected components
    labeled_image, num_features = scipy.ndimage.label(binary_image)

    # Create a boolean array to identify border touching components
    border_touching = np.zeros(labeled_image.shape, dtype=bool)

    # Check the borders: first and last rows and first and last columns
    border_touching[0, :] = labeled_image[0, :] > 0
    border_touching[-1, :] = labeled_image[-1, :] > 0
    border_touching[:, 0] = labeled_image[:, 0] > 0
    border_touching[:, -1] = labeled_image[:, -1] > 0

    # Find the labels of the components that touch the border
    touching_labels = np.unique(labeled_image[border_touching])
    touching_labels = touching_labels[touching_labels > 0]  # Ignore background (0)

    # Create a mask to keep only the components that touch the border
    mask = np.isin(labeled_image, touching_labels)

    # Apply the mask to the binary image
    filtered_image = mask.astype(int)

    return filtered_image



def box_count(curve, box_size):
    """Counts the number of boxes needed to cover the curve."""
    min_x, min_y = np.min(curve, axis=0)
    max_x, max_y = np.max(curve, axis=0)

    # Create a grid of boxes
    x_edges = np.arange(min_x, max_x + box_size, box_size)
    y_edges = np.arange(min_y, max_y + box_size, box_size)

    # Count boxes
    count = 0
    for x in x_edges:
        for y in y_edges:
            box = [x, x + box_size, y, y + box_size]
            # Check if the box intersects with the curve
            if np.any((curve[:, 0] >= box[0]) & (curve[:, 0] < box[1]) &
                       (curve[:, 1] >= box[2]) & (curve[:, 1] < box[3])):
                count += 1
    return count

import numpy as np
import matplotlib.pyplot as plt

def fractal_dimension(curve, box_sizes, plot=False):
    """Calculates the fractal dimension of a curve using the box-counting method."""
    counts = [box_count(curve, size) for size in box_sizes]
    log_sizes = np.log(box_sizes)
    log_counts = np.log(counts)

    # Perform a linear fit to find the slope
    coeffs = np.polyfit(log_sizes[log_counts > 0], log_counts[log_counts > 0], 1)

    dimension = -coeffs[0]

    if plot:
        plt.figure()
        plt.scatter(log_sizes[log_counts > 0], log_counts[log_counts > 0], marker='o', 
                    color='black', label='Data points', alpha=0.8, s=10)
        
        # Calculate the fitted line
        fit_line = coeffs[0] * log_sizes + coeffs[1]
        
        # Plot the regression line
        plt.plot(log_sizes[log_counts > 0], fit_line[log_counts > 0], color='red', linestyle='--', label='Regression line')

        plt.title('Regression to get fractal dimension: d={:.3f}'.format(dimension))
        plt.xlabel('log(box size)')
        plt.ylabel('log(box counts)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return dimension
