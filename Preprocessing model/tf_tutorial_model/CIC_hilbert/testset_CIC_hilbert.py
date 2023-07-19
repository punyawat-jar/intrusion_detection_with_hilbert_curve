import os
import matplotlib.pyplot as plt
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import pickle
import glob
import requests, json
import multiprocessing
from multiprocessing import Value
from functools import partial
from IPython.display import clear_output
import ctypes
from PIL import Image

def send_discord_message(content):
    webhook_url = 'https://discord.com/api/webhooks/1120592726648434838/qZejSK_KnJymtR8mN6uaXiouf1cK2GFgOTuPHbbd7TRGja3gOr4KcBAGxbJ_MAa93NP8'

    data = {
        'content': content
    }

    response = requests.post(webhook_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

    if response.status_code != 204:
        raise ValueError(f'Request to discord returned an error {response.status_code}, the response is:\n{response.text}')

def read_DB():
    dftrain = pd.read_csv('/home/s2316002/Project2/Dataset/CIC/Cleaned_balanced.csv')
    dftrain.loc[dftrain['label'] == "BENIGN", "label"] = 0
    dftrain.loc[dftrain['label'] != 0, "label"] = 1
    return dftrain

def preprocess(df):
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df
def image_show(image,column):
    print(f'Column = {column}, size = {image.shape}')
    plt.figure(figsize=(5, 5))  # Set the figure size
    plt.imshow(image, cmap = 'gray', origin='lower')  # Use 'gray' colormap for grayscale
    plt.grid(True)
    plt.show()

def add_row_padding(df, padding):
    dfs = []  # list to hold DataFrames

    # padding DataFrame
    padding_df = pd.DataFrame(np.zeros((padding, df.shape[1])), columns=df.columns)

    # iterate over rows in the original DataFrame
    for idx in range(df.shape[0]):
        # add the row and the padding to our list
        dfs.extend([df.iloc[[idx]], padding_df])

    # concatenate all the DataFrames in our list
    result = pd.concat(dfs, ignore_index=True)

    # remove padding after the last row of original dataframe
    result = result.iloc[:-padding]

    return result

## Generate Hilbert's Curve
def hilbert_curve(order):
    points = [(0, 0), (0, 1), (1, 1), (1, 0)]

    def transform(s, dx, dy, rot):
        if rot == 0:
            return dx, dy
        else:
            return dy, 2*s-dx-1 if (rot == 1 or rot == 3) else dx, 2*s-dy-1 if (rot == 2 or rot == 3) else dy

    def generate_curve(order, rot, s):
        if order == 0:
            return [points[rot]]
        else:
            order -= 1
            s //= 2
            return generate_curve(order, (rot+1)%4, s) + [(x+s, y+s) for (x, y) in generate_curve(order, rot, s)] + generate_curve(order, (rot+1)%4, s) + [(x+s, y+3*s) for (x, y) in generate_curve(order, (rot+3)%4, s)] + generate_curve(order, (rot+2)%4, s)

    return generate_curve(order, 0, 2**order)

def load_from_file(filename):
    coordinates = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            coord_str = line.strip('()\n').split(',')  # Split on space
            coordinates.append((int(coord_str[0]), int(coord_str[1])))
    return coordinates

def last_two_bits(x):
    """Return the last two bits of a number."""
    return x & 3

def hindex_to_xy(hindex, N):
    """Convert a Hilbert curve index to a pair of coordinates."""
    positions = [(0, 0), (0, 1), (1, 1), (1, 0)]

    temp = positions[last_two_bits(hindex)]
    hindex >>= 2
    x, y = temp
    n = 4
    while n <= N:
        n2 = n // 2
        pos_in_small_square = last_two_bits(hindex)

        if pos_in_small_square == 0:  # lower left
            x, y = y, x
        elif pos_in_small_square == 1:  # upper left
            y += n2
        elif pos_in_small_square == 2:  # upper right
            x += n2
            y += n2
        elif pos_in_small_square == 3:  # lower right
            x, y = (n2 - 1) - y, (n2 - 1) - x
            x += n2

        hindex >>= 2
        n *= 2

    return x, y

def generate_and_save_hilbert(order, file_name):
    """Generate Hilbert curve coordinates and save to a file."""
    N = 2**order
    point_list = [(hindex_to_xy(i, N)) for i in range(N*N)]

    print("Writing to file...")
    with open(file_name, "w") as pixel_file:
        for point in point_list:
            pixel_file.write(str(point) + '\n')

    print(f"Pixel count: {len(point_list)}")
    return point_list

def draw_hilbert(coordinates, fig_width, fig_height):
    """Draw a Hilbert curve from a list of coordinates."""
    pad = 1
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_width, fig_height)
    ax.set_aspect('equal', 'box')  # ensure equal aspect ratio
    ax.axis('on')  # remove axes for better visualization

    min_x = min(coord[0] for coord in coordinates) - pad
    max_x = max(coord[0] for coord in coordinates) + pad
    min_y = min(coord[1] for coord in coordinates) - pad
    max_y = max(coord[1] for coord in coordinates) + pad
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    print("Drawing Hilbert curve...")
    prev_point = coordinates[0]
    for i in range(1, len(coordinates)):
        curr_point = coordinates[i]
        line_x, line_y = zip(prev_point, curr_point)
        ax.add_line(Line2D(line_x, line_y, linewidth=1.5, color='blue'))  # Increased linewidth
        prev_point = curr_point

        if i % 1000 == 0:
            print(f"Processed {i} points")

    plt.show()

def split_label(df):
    x = df.drop(['label'], axis = 1)
    y = df.label
    return x, y

def check_duplicates(lst):
    duplicates = []
    count_dict = {}

    for item in lst:
        if isinstance(item, list):
            item = tuple(item)  # Convert lists to tuples

        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1

    for item, count in count_dict.items():
        if count > 1:
            duplicates.append(item)

    return duplicates


def make_image(df, cor, size):
    image = np.zeros((size, size, df.shape[1]))

    for i in range(df.shape[0]):
        for j, feature in enumerate(df.columns):
            x, y = cor[i]
            image[x][y][j] = df.iloc[i][feature]

    return image

def plot_image(image):
    for i in range(image.shape[2]):
        plt.imshow(image[:, :, i], cmap='gray')  # Assuming grayscale image
        plt.title(f"Layer '{df.columns[i]}'")
        plt.show()

def drop_feature(df, feature):
    df = df.drop(feature, axis = 1)
    return df


# def concat_image(image, x, y, size):
#     output_size_X = x*size
#     output_size_y = y*size
    
#     # First we transpose the dimensions so the channels are first
#     image = np.transpose(image, (2, 0, 1))

#     # Now we reshape the array so the rows and columns are split into subgrids
#     image = image.reshape((10, 12, 8, 8))

#     # Reorder so that it's 10 rows each containing 12 images of 8x8
#     image = image.transpose(0, 2, 1, 3)

#     # Reshape into 2D grid
#     image_2d = image.reshape(80, 96)
#     return image_2d

def concat_image(image, x, y, size):
    # Determine the output size
    output_size_X = x * size
    output_size_Y = y * size

    # Assume image is your initial image
    # First we transpose the dimensions so the channels are first
    image = np.transpose(image, (2, 0, 1))

    # Now we reshape the array so the rows and columns are split into subgrids
    image = image.reshape((y, x, size, size))

    # Reorder so that it's y rows each containing x images of size x size
    image = image.transpose(0, 2, 1, 3)

    # Reshape into 2D grid
    concatenated_image = image.reshape(output_size_X, output_size_Y)

    return concatenated_image


def generate_df_slices(df, window_size):
    temp_df_list = []
    for i in range(len(df) - window_size + 1):
        temp_df = df[i:i+window_size]
        temp_df_list.append(temp_df)
    return temp_df_list
    
def process_labels(temp_df_list):
    labels = [split_label(df_slice)[1].values[-1] for df_slice in temp_df_list]
    return labels

def process_and_save_image(i, df_slice, loaded_curve, drop_fea, size, n_windows):
    # Drop the specified features
    
    # print('enter process and save')
    df_slice = drop_feature(df_slice, drop_fea)
    # print('drop_feature successful')
    # clear_output(wait=True)
    # Split the DataFrame into features and label
    tempx, _ = split_label(df_slice)  # _ means we ignore the label here
    # print('spllit_label_successful')
    # Make an image from the feature data
    image = make_image(tempx, loaded_curve, size)
    # print('makeimage successful')
    # Concatenate the image layers
    concatenated_image = concat_image(image, 11, 7, size) ###  11, 7 is the size of the function 
    # print('concat successful')
    # # Save the image to a file
    # plt.imshow(concatenated_image, cmap='gray')
    # plt.axis('off')  # Turn off axes
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Eliminate margins
    # plt.savefig(f'/home/s2316002/Project2/Dataset/CIC/Image/image_cic_{i}.png', bbox_inches='tight', pad_inches=0)
    # plt.close()
    # print(f'Image {i}/{n_windows} -- Done')
    
    # Save the image to a file
    img = Image.fromarray((concatenated_image * 255).astype(np.uint8))  # Convert the image to uint8 before saving
    img.save(f'/home/s2316002/Project2/Dataset/CIC/image_test/image_{i}.png')
    print(f'Image {i}/{n_windows} -- Done')

    with counter.get_lock():
        counter.value += 1
    
    if counter.value % 5000 == 0 or counter.value <2:
        progress = counter.value / n_windows
        progress_bar = '[' + '#' * int(progress * 20) + '-' * int((1 - progress) * 20) + ']'
        content = f'-testset - Finished processing image {counter.value}. Progress: {progress * 100:.2f}% \n{progress_bar}'
        send_discord_message(content)
    return df_slice

    # read_DB() # read new data
df = read_DB()
df = preprocess(df)
divided = df.shape[0]/2
train = df[:int(divided)]
# y_train = df[:int(divided)].label
test = df[int(divided):]
# y_test = df[int(divided):].label
    
# order = int(input("Enter the order of the Hilbert curve: "))
order = 4
size = 2**order
filename = "/home/s2316002/Project2/hilbert_curve_order4.txt"
curve = generate_and_save_hilbert(order, filename)
counter = Value(ctypes.c_int, 0)
    # save_to_file(filename, curve)
    
loaded_curve = load_from_file(filename)
    
window_size = size * size
n_windows = df.shape[0] - window_size - 1
drop_fea = ['bwd_avg_bulk_rate']
    
# Determine the starting point based on existing files
image_files = glob.glob('/home/s2316002/Project2/Dataset/CIC/image_test/image_*.png')
send_discord_message(f'- CIC Hillbert starting -')
print(f'- CIC Hillbert starting (testset) -')
    
    # Prepare function for multiprocessing
process_func = partial(
    process_and_save_image,
    loaded_curve=loaded_curve,
    drop_fea=drop_fea,
    size=size,
    n_windows=n_windows,
)
send_discord_message('process_func done')
print('process_func done')
temp_df_list = generate_df_slices(train, window_size)
send_discord_message('generate slide done')
print('generate slide done')
    # Use multiprocessing
with multiprocessing.Pool() as pool:
    processed_slices = pool.starmap(process_func, enumerate(temp_df_list))
send_discord_message('multiprocess pool done')
print('multiprocess pool done')
    
labels = process_labels(temp_df_list)
with open('labels2.pkl', 'wb') as f:
    pickle.dump(labels, f)
        
send_discord_message('All Process is Done!!!')
