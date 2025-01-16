import matplotlib.pyplot as plt
import matplotlib.image as mpimg


if __name__ == "__main__":

    img1 = mpimg.imread('demo2_part1.jpg')
    img2 = mpimg.imread('demo2_part2.jpg')
    img3 = mpimg.imread('demo2_part3.jpg')


    # Create a figure and set up subplots with 3 rows and 1 column
    fig, axes = plt.subplots(3, 1, figsize=(5, 10))  # 3 rows, 1 column

    # Display images in each subplot
    axes[0].imshow(img1)
    axes[0].axis('off')  # Hide axes for a cleaner look
    axes[0].set_title('Image 1')

    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Image 2')

    axes[2].imshow(img3)
    axes[2].axis('off')
    axes[2].set_title('Image 3')

    # Add horizontal lines between each row
    for ax in axes:  # Loop through the first two axes (before the last row)
        ax.axhline(y=-5, color='black', linewidth=2)  # Adjust 'y' to position the line

    # Adjust layout to prevent overlap and add spacing between rows
    plt.subplots_adjust(hspace=0.5)

    plt.savefig('images_table_with_lines.jpg', format='jpg', dpi=300)

    plt.show()