import numpy as np
import matplotlib.pyplot as plt

class Tracker(object):
    def __init__(self, ax, CT):
        self.ax = ax
        ax.set_title('CT Visualization')
        self.CT = CT
        self.slices = CT.shape[0]
        self.ind = self.slices//2
        self.im = ax.imshow(self.CT[self.ind, :, :], cmap='gray')
        self.update() #upon instatiation, draw the image at self.ind on the axes' canvas

    def scroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices #modulus, so when we reach the end of the patient, we start over
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.CT[self.ind, :, :]) #update image data
        self.ax.set_ylabel('Slice {}'.format(self.ind)) #update slice label
        self.im.axes.figure.canvas.draw() #draw new data on axes
        
def show_CT(ct_file):
    ct = np.load(ct_file)
    fig, axs = plt.subplots()
    tracker = Tracker(axs, ct)
    fig.canvas.mpl_connect('scroll_event', tracker.scroll)
    plt.show()
    
if __name__ == '__main__':
    show_CT('ct.npy')