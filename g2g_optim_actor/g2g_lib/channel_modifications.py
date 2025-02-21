import numpy as np

# Return an array of the size of the channels
# Each element is the grid channel index
# no_grid_index    : Number of of channel slice in the cube
# grid_channel_idx : array of index to convert the channel number to the grid slice index
def set_grid_channel_index(no_spw, no_chan, channels_to_image):
    grid_channel_idx = np.zeros(no_spw*no_chan, dtype=np.int32)

    # For now channels_to_image is bool
    # TODO channels_to_image is a tuple of channels to be imaged
    no_grid_index = 0
    grid_channel_width = 0
    if channels_to_image is True:
        no_grid_index = no_spw*no_chan
        grid_channel_idx = np.arange(no_spw*no_chan, dtype=np.int32)
    else:
        no_grid_index = 1

    grid_channel_width = no_chan/no_grid_index

    return (grid_channel_idx, no_grid_index, int(grid_channel_width))

