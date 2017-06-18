import tkinter as tk
from PIL import Image, ImageTk
import math
import urllib.request


def deg2num(lat_deg, lon_deg, zoom):
    """ Lon./lat. to tile numbers
    From https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid(padx=10, pady=10)

        # Set up mode state
        self.current_mode = tk.IntVar()
        self.current_mode.set(2)  # initialize
        self.current_image = None

        # Create marker dict
        self.marker_dict = dict()
        self.loc = (tk.StringVar(), tk.StringVar(), tk.IntVar())
        self.loc[0].set('15.0')
        self.loc[1].set('-13.0')
        self.loc[2].set(18)

        self.createWidgets()
        self.showImage()

    def createWidgets(self):

        # Make widgets
        # Coordinate entry
        coord_box = tk.LabelFrame(self, text='Location', width=256, labelanchor='n')
        lat = tk.Entry(coord_box, textvariable=self.loc[0])
        long = tk.Entry(coord_box, textvariable=self.loc[1])
        zl = tk.Entry(coord_box, textvariable=self.loc[2])

        lat.grid(row=0, column=1)
        long.grid(row=1, column=1)
        zl.grid(row=2, column=1)

        lat_label = tk.Label(coord_box, text='Latitude')
        lat_label.grid(row=0, column=0)
        long_label = tk.Label(coord_box, text='Longitude')
        long_label.grid(row=1, column=0)
        zoom_label = tk.Label(coord_box, text='Zoom Level')
        zoom_label.grid(row=2, column=0)


        fetch_button = tk.Button(coord_box, text='Fetch image', command=self.fetchImage)
        fetch_button.grid(row=3, columnspan=2)

        # Mode radio buttons
        mode_box = tk.LabelFrame(self, text='Mode', width=256, labelanchor='n')

        modes = [
            ("Label", 2),
            ("Delete", 0),
            ("Maybe", 1)
        ]
        for text, mode in modes:
            b = tk.Radiobutton(mode_box, text=text,
                               variable=self.current_mode, value=mode, indicatoron=0)
            b.grid(row=0, column=mode)

        # Row 2 separator
        separator = tk.Frame(self, height=5)

        # Row 3 canvas
        self.canvas = tk.Canvas(self, width=256, height=256)
        self.canvas.bind('<Button-1>', self.editLabel)

        # Row 4-5 coordinate list
        coord_title = tk.Label(self, text='Labels')
        self.coord_list = tk.Listbox(self, width=30)

        # Row 6 save coordinates / next image
        nav_box = tk.Frame(self)
        prev_button = tk.Button(nav_box, text='Back', command=self.prevImage)
        save_button = tk.Button(nav_box, text='Save', command=self.saveCoords)
        next_button = tk.Button(nav_box, text='Next', command=self.nextImage)

        prev_button.grid(row=1, column=0)
        save_button.grid(row=1, column=1)
        next_button.grid(row=1, column=2)

        # Order widgets
        coord_box.grid(row=0, column=1)
        mode_box.grid(row=1, column=1)
        separator.grid(row=2, column=1)
        self.canvas.grid(row=3, column=1)
        coord_title.grid(row=4, column=1)
        self.coord_list.grid(row=5, column=1)
        nav_box.grid(row=6, column=1)

    def fetchImage(self):
        lat = float(self.loc[0].get())
        long = float(self.loc[1].get())
        zl = int(self.loc[2].get())
        # zoom = 18
        x, y = deg2num(lat, long, zl)
        path = "https://mt2.google.com/vt/lyrs=s&x={}&y={}&z={}".format(x, y, zl)
        print(path)
        local_filename, _ = urllib.request.urlretrieve(path)
        print(local_filename)
        self.showImage(local_filename)


    def showImage(self, path='sample.jpeg'):
        # need to tag this as 1
        photo = ImageTk.PhotoImage(Image.open(path))
        self.current_image = self.canvas.create_image(0, 0, anchor='nw', image=photo)
        self.canvas.current_image = photo

    def editLabel(self, event):
        # Specify size of marker
        rad = 5
        x1, y1 = (event.x - rad), (event.y - rad)
        x2, y2 = (event.x + rad), (event.y + rad)
        closest_marker = self.canvas.find_closest(event.x, event.y)

        if self.current_mode.get() == 0:
            # Remove annotation from canvas
            self.canvas.delete(closest_marker)

            # Remove annotation from visible list
            self.coord_list.delete(self.marker_dict[closest_marker[0]])

            # Remove entry from backend dictionary
            self.marker_dict.pop(closest_marker[0])


        elif self.current_mode.get() == 1:
            # If the closest marker is the canvas itself, add red
            if closest_marker[0] == self.current_image:
                marker = self.canvas.create_oval(x1, y1, x2, y2, fill='red', tags=('marker',))
                label_str = self.getCoords(event, marker) + ' (maybe)'
                self.coord_list.insert(tk.END, label_str)
            # if in the space of the dot, make red
            else:
                marker = self.canvas.itemconfig(closest_marker, fill='red')
                self.coord_list.delete(self.marker_dict[closest_marker[0]])
                label_str = self.getCoords(event, marker) + ' (maybe)'
                self.coord_list.insert(self.marker_dict[closest_marker[0]], label_str)
        else:
            # Create dot
            marker = self.canvas.create_oval(x1, y1, x2, y2, fill='blue', tags=('marker',))
            label_str = self.getCoords(event, marker)
            self.coord_list.insert(tk.END, label_str)

    def getCoords(self, event, marker):
        # Find coordinates in image and print to table
        self.marker_dict[marker] = self.coord_list.size()
        canvas = event.widget
        x = canvas.canvasx(event.x)
        y = canvas.canvasy(event.y)
        label_str = '{},{}'.format(x,y)
        return label_str

    def saveCoords(self):
        # Get coordinates of all labels and write to file as metadata for image tile
        # Writes to file with timestamp (take latest)
        print(self.coord_list.get(0, tk.END))

    def nextImage(self):
        # delete markers using marker dictionary keys
        for k in self.marker_dict.keys():
            self.canvas.delete(k)

        # delete coordinates
        self.coord_list.delete(0, tk.END)

        # clear marker dictionary
        self.marker_dict = dict()

        self.showImage('sample_airplanes.jpeg')

    def prevImage(self):
        return None




app = Application()
app.master.title('Termite Mound Labeller')
app.mainloop()


