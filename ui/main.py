import tkinter as tk
from PIL import Image, ImageTk
import math
from random import randint
import urllib.request
from time import time



def deg2num(lat_deg, lon_deg, zoom):
    """ Lon./lat. to tile numbers
    From https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

class DataCollector(object):
    def __init__(self):
        """
        Metadata to collect during data collection
            * Time stamp
            * Tile numbers (x, y, z)
            * Coordinates (lat_deg, lon_deg)
            * Annotations

        Image tile dictionary
        Key is the tuple (x, y, z)
        Nested dictionary of visit number
        Elements are dictionary with:
            * URL
            * URL fetch time
            * HTTP headers
            * Local filepath
            * Labels (dictionary)
        """

        self.data = dict()


    def createEntry(self, tile_tuple, fetch_url, headers, local_filename):

        self.current_tile = tile_tuple

        if tile_tuple in self.data.keys():
            self.data[tile_tuple]['visit_count'] += 1
        else:
            self.data[tile_tuple] = {'visit_count': 0}

        visit_no = self.data[tile_tuple]['visit_count']

        self.current_visit = visit_no

        self.data[tile_tuple][visit_no] = {
            'url': fetch_url,
            'response': headers,
            'local': local_filename
        }

    def updateEntry(self, tile_tuple, label_list):
        visit_no = self.data[tile_tuple]['visit_count']
        self.data[tile_tuple][visit_no]['labels'] = label_list
        self.data[tile_tuple][visit_no]['tstamp'] = time()

    def print(self):
        # print(self.data)
        with open('data.csv', mode='a') as print_to:
            for tile in self.data.keys():
                visit_count = self.data[tile]['visit_count']
                for visit in range(visit_count + 1):
                    this = self.data[tile][visit]
                    labels = this['labels']
                    for label in labels:
                        print(*tile, visit, visit_count, this['tstamp'], label,
                              this['url'], this['local'], sep=',', file=print_to, flush=True)


class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid(padx=10, pady=10)

        # Set up tile queue and data collector
        self.Queue = list()
        self.fillQueue()
        self.current_tile = 0
        self.Data = DataCollector()

        # Set up mode state
        self.current_mode = tk.IntVar()
        self.current_mode.set(2)  # initialize
        self.current_image = None

        # Create marker dict
        self.marker_dict = dict()
        self.loc = dict(
            lat=tk.StringVar(),
            lon=tk.StringVar(),
            zl=tk.IntVar(),
            xtile=0,
            ytile=0
        )

        self.setLocationFromQueue()
        self.createWidgets()
        self.updateImageFromTileNumbers()

    def fillQueue(self, base_lat=15.0, base_lon=-13.0, zl=18):
        # base_lat = 15.0
        # base_lon = -13.0
        # zl = 18
        for i in range(-5, 5, 1):
            for j in range(-5, 5, 1):
                lat = base_lat + (i*0.01)
                lon = base_lon + (j*0.01)
                x, y = deg2num(lat, lon, zl)
                self.Queue.append((lat, lon, zl, x, y))

    def setLocationFromQueue(self):
        lat, lon, zl, x, y = self.Queue[self.current_tile]
        self.loc['lat'].set(str(lat))
        self.loc['lon'].set(str(lon))
        self.loc['zl'].set(zl)
        self.loc['xtile'], self.loc['ytile'] = x, y

    def createLabelWidgets(self):
        self.label_box = tk.Frame(self)
        # Mode radio buttons
        mode_box = tk.LabelFrame(self.label_box, text='Mode', width=256, labelanchor='n')

        modes = [
            ("Label", 2),
            ("Delete", 0),
            ("Maybe", 1)
        ]
        for text, mode in modes:
            b = tk.Radiobutton(mode_box, text=text,
                               variable=self.current_mode, value=mode, indicatoron=0)
            b.grid(row=0, column=mode)

        # Coordinate list
        coord_title = tk.Label(self.label_box, text='Labels')
        self.coord_list = tk.Listbox(self.label_box, width=30)

        # Nav box: Save coordinates / next image
        nav_box = tk.Frame(self.label_box)
        prev_button = tk.Button(nav_box, text='Back', command=self.prevImage)
        save_button = tk.Button(nav_box, text='Save', command=self.saveCoords)
        next_button = tk.Button(nav_box, text='Next', command=self.nextImage)
        prev_button.grid(row=1, column=0)
        save_button.grid(row=1, column=1)
        next_button.grid(row=1, column=2)

        show_button = tk.Button(nav_box, text='Show queue', command=self.showQueue)
        show_button.grid(row=2, columnspan=3)

        # Title
        label_title = tk.Label(self.label_box, text='Label', fg='blue')
        label_title.grid(row=0, columnspan=3)

        # Place labelling components
        mode_box.grid(row=1, columnspan=3)
        coord_title.grid(row=2, columnspan=3)
        self.coord_list.grid(row=3, columnspan=3)
        nav_box.grid(row=4, columnspan=3)

    def createDiscoverWidgets(self):

        self.discover_box = tk.Frame(self)

        # Coordinate entry
        coord_box = tk.LabelFrame(self.discover_box, text='Location', width=256, labelanchor='n')
        lat = tk.Entry(coord_box, textvariable=self.loc['lat'])
        lon = tk.Entry(coord_box, textvariable=self.loc['lon'])
        zl = tk.Entry(coord_box, textvariable=self.loc['zl'])

        # Coord_box
        lat.grid(row=0, column=1)
        lon.grid(row=1, column=1)
        zl.grid(row=2, column=1)

        lat_label = tk.Label(coord_box, text='Latitude')
        lat_label.grid(row=0, column=0)
        lon_label = tk.Label(coord_box, text='Longitude')
        lon_label.grid(row=1, column=0)
        zoom_label = tk.Label(coord_box, text='Zoom Level')
        zoom_label.grid(row=2, column=0)

        fetch_button = tk.Button(coord_box, text='Fetch image', command=self.loadImageFromCoords)
        fetch_button.grid(row=3, columnspan=2)

        # Compass box
        cps_box = tk.LabelFrame(self.discover_box, text='Compass', labelanchor='n', width=256)
        n_button = tk.Button(cps_box, text='North', command=self.goNorth)
        s_button = tk.Button(cps_box, text='South', command=self.goSouth)
        w_button = tk.Button(cps_box, text='West', command=self.goWest)
        e_button = tk.Button(cps_box, text='East', command=self.goEast)
        n_button.grid(row=0, column=1)
        w_button.grid(row=1, column=0)
        e_button.grid(row=1, column=2)
        s_button.grid(row=2, column=1)

        # Title
        discover_title = tk.Label(self.discover_box, text='Discover', fg='blue')
        discover_title.grid(row=0, column=1)

        # Place discover components
        coord_box.grid(row=1, column=1)
        cps_box.grid(row=2, column=1)

    def createWidgets(self):

        # Make widgets
        self.createLabelWidgets()
        self.createDiscoverWidgets()

        separator = tk.Frame(self, height=5)
        # Row 3 canvas
        self.canvas = tk.Canvas(self, width=256, height=256)
        self.canvas.bind('<Button-1>', self.editLabel)

        # separator.grid(row=1, column=1)
        self.canvas.grid(row=1, column=1)
        self.label_box.grid(row=2, column=1)
        self.discover_box.grid(row=3, column=1)

    def goNorth(self):
        self.loc['ytile'] -= 1
        self.updateImageFromTileNumbers()

    def goSouth(self):
        self.loc['ytile'] += 1
        self.updateImageFromTileNumbers()

    def goEast(self):
        self.loc['xtile'] += 1
        self.updateImageFromTileNumbers()

    def goWest(self):
        self.loc['xtile'] -= 1
        self.updateImageFromTileNumbers()

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
                label_str = self.getLabelString(event, marker) + ' (maybe)'
                self.coord_list.insert(tk.END, label_str)
            # if in the space of the dot, make red
            else:
                marker = self.canvas.itemconfig(closest_marker, fill='red')
                self.coord_list.delete(self.marker_dict[closest_marker[0]])
                label_str = self.getLabelString(event, marker) + ' (maybe)'
                self.coord_list.insert(self.marker_dict[closest_marker[0]], label_str)
        else:
            # Create dot
            marker = self.canvas.create_oval(x1, y1, x2, y2, fill='blue', tags=('marker',))
            label_str = self.getLabelString(event, marker)
            self.coord_list.insert(tk.END, label_str)

    def showQueue(self):
        """
        Display
        - coordinates
        - tile numbers, zoom level
        - number of labels (total)
        - number of definite/maybe labels
        - expand to show individual locations?
        :return:
        """
        self.data_disp = tk.Toplevel(width=256)
        self.data_disp.title('Queue')
        msg = tk.Message(self.data_disp, text='Current tile is {}'.format(self.current_tile), width=256)
        msg.grid(row=0, column=0)
        tile_list = tk.Listbox(self.data_disp)
        tile_list.grid(row=1, column=0)
        for i, item in enumerate(self.Queue):
            tile_list.insert(tk.END, '{}\t{}\n'.format(i, item))
        tile_list.activate(self.current_tile)


    def showImage(self, path='sample.jpeg'):
        # need to tag this as 1
        # self.updateCoordsFromCurrentTile()
        photo = ImageTk.PhotoImage(Image.open(path))
        self.current_image = self.canvas.create_image(0, 0, anchor='nw', image=photo)
        self.canvas.current_image = photo  # maintain reference to prevent garbage collection

    def updateImageFromTileNumbers(self):
        path = self.downloadImage(self.loc['xtile'], self.loc['ytile'], self.loc['zl'].get())
        self.showImage(path)
        self.updateCoordsFromCurrentTile()

    def downloadImage(self, x, y, zl):
        n = randint(0, 3)
        path = "https://mt{}.google.com/vt/lyrs=s&x={}&y={}&z={}".format(n, x, y, zl)
        local_filename, headers = urllib.request.urlretrieve(path)
        self.Data.createEntry((x, y, zl), path, headers, local_filename)
        return local_filename

    def loadImageFromCoords(self):
        lat = float(self.loc['lat'].get())
        lon = float(self.loc['lon'].get())
        zl = int(self.loc['zl'].get())
        x, y = deg2num(lat, lon, zl)
        self.loc['xtile'] = x
        self.loc['ytile'] = y
        self.updateImageFromTileNumbers()

    def updateCoordsFromCurrentTile(self):
        lat_deg, lon_deg = num2deg(self.loc['xtile'], self.loc['ytile'], self.loc['zl'].get())
        self.loc['lat'].set(str(lat_deg))
        self.loc['lon'].set(str(lon_deg))

    def getLabelString(self, event, marker):
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
        tile_tuple = (self.loc['xtile'], self.loc['ytile'], self.loc['zl'].get())
        label_list = self.coord_list.get(0, tk.END)
        self.Data.updateEntry(tile_tuple, label_list)
        print(label_list)
        self.Data.print()

    def clearCanvas(self):
        # delete markers using marker dictionary keys
        for k in self.marker_dict.keys():
            self.canvas.delete(k)

        # delete coordinates
        self.coord_list.delete(0, tk.END)

        # clear marker dictionary
        self.marker_dict = dict()

    def nextImage(self):
        self.clearCanvas()
        self.current_tile += 1
        self.setLocationFromQueue()
        self.updateImageFromTileNumbers()

    def prevImage(self):
        self.clearCanvas()
        self.current_tile -= 1
        self.setLocationFromQueue()
        self.updateImageFromTileNumbers()





app = Application()
app.master.title('Labeller')
app.mainloop()


