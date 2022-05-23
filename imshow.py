def imshow(img, title):

    """Custom function to display the image using matplotlib"""

    #define std correction to be made
    std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    #define mean correction to be made
    mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)

    #convert the tensor img to numpy img and de normalize 
    npimg = np.multiply(img.numpy(), std_correction) + mean_correction

    #plot the numpy image
    plt.figure(figsize = (batch_size * 4, 4))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()