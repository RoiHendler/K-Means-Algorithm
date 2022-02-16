import matplotlib.pyplot as plt
import numpy as np
import sys


MAX_ITERATIONS = 20


def distance(a,b):
    return np.linalg.norm(a-b)


def closest_pixel(pixel, centroids):
    # return the index of the closest centroid to the given pixel
    distances = []
    for centroid in centroids:
        distances.append(distance(pixel, centroid))
    distances = np.array(distances)
    return distances.argmin(), distances.min()


def k_means(pixels, centroids):
    output_str = ""
    losses = []

    for iteration in range(MAX_ITERATIONS):
        #each element corresponds by index to a pixel, and specifies its centroid assignment by centroid index
        classifications = []
        total_distance = 0
        for pixel_idx in range(len(pixels)):
            centroid, distance = closest_pixel(pixels[pixel_idx], centroids)
            classifications.append(centroid)
            total_distance += distance
        losses.append(total_distance)

        #iterate all centroids
        #for each centroid find all assigned pixels
        #calculate mean for assigned pixels
        #replaces old centroids with the new mean of each one
        #stop when no change occured
        new_centroids = []
        for centroid_idx in range(len(centroids)):
            centroid_pixels = []
            for idx, classification in enumerate(classifications):
                if classification == centroid_idx:
                    centroid_pixels.append(pixels[idx])
            if len(centroid_pixels) == 0:
                continue
            centroid_pixels = np.array(centroid_pixels)
            centroid_avg = centroid_pixels.mean(axis=0).round(4)
            new_centroids.append(centroid_avg)

        output_str += f"[iter {iteration}]:{','.join([str(i) for i in new_centroids])}\n"

        new_centroids = np.array(new_centroids)
        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, output_str, losses


def main(plot=False):
    image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
    centroids = np.loadtxt(centroids_fname)
    number_of_centroids = len(centroids)

    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float)/255
    pixels_shape = pixels.shape
    pixels = pixels.reshape(-1,3)

    centroids, output_str, losses = k_means(pixels, centroids)

    with open(out_fname, "w") as f:
        f.write(output_str)

    if plot:
        plt.plot(losses)
        plt.xlabel("iterations")
        plt.ylabel("total loss")
        plt.title(f"loss with k={number_of_centroids}")
        plt.show()

        # Display the pictures
        new_pixels = []
        for pixel in pixels:
            new_pixels.append(centroids[closest_pixel(pixel, centroids)[0]])

        new_pixels = np.array(new_pixels).reshape(pixels_shape)

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(orig_pixels)
        axarr[1].imshow(new_pixels)
        plt.title(f"k={number_of_centroids}")
        plt.show()


main()
