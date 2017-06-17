import tkinter as tk
from PIL import Image, ImageTk

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid(padx=10, pady=10)
        self.currentMode = tk.IntVar()
        self.currentMode.set(2)  # initialize
        self.createWidgets()
        self.showImage()
        self.marker_dict = dict()

    def activate(self):
        print(self.currentMode.get())


    def createWidgets(self):

        # Row 1
        # Mode radio buttons
        self.modeBox = tk.LabelFrame(self, text='Mode', width=256, labelanchor='n')
        self.modeBox.grid(row=1, column=1)

        MODES = [
            ("Label", 2),
            ("Delete", 0),
            ("Maybe", 1)
        ]
        for text, mode in MODES:
            b = tk.Radiobutton(self.modeBox, text=text,
                               variable=self.currentMode, value=mode, indicatoron=0, command=self.activate)
            b.grid(row=0, column=mode)

        # Row 2 separator
        separator = tk.Frame(self, height=5)
        separator.grid(row=2, column=1)

        # Row 3 canvas
        self.canvas = tk.Canvas(self, width=256, height=256)
        self.canvas.grid(row=3, column=1)
        self.canvas.bind('<Button-1>', self.editLabel)


        # Set up label listbox
        self.coordTitle = tk.Label(self, text='Labels')
        self.coordTitle.grid(row=4, column=1)
        self.coordList = tk.Listbox(self, width=30)
        self.coordList.grid(row=5, column=1)

    def showImage(self):
        photo = ImageTk.PhotoImage(Image.open('sample.jpeg'))
        self.canvas.create_image(0, 0, anchor='nw', image=photo)
        self.canvas.current_image = photo

    def editLabel(self, event):

        # Specify size of marker
        rad = 5
        x1, y1 = (event.x - rad), (event.y - rad)
        x2, y2 = (event.x + rad), (event.y + rad)

        closest_marker = self.canvas.find_closest(event.x, event.y)

        if self.currentMode.get() == 0:
            # Remove annotation from canvas
            self.canvas.delete(closest_marker)

            # Remove annotation from visible list
            self.coordList.delete(self.marker_dict[closest_marker[0]])

            # Remove entry from backend dictionary
            self.marker_dict.pop(closest_marker[0])


        elif self.currentMode.get() == 1:

            # If the closest marker is the canvas itself, add red
            if closest_marker[0] == 1:
                marker = self.canvas.create_oval(x1, y1, x2, y2, fill='red')
                label_str = self.getCoords(event, marker) + ' (maybe)'
                self.coordList.insert(tk.END, label_str)
            # if in the space of the dot, make red
            else:
                marker = self.canvas.itemconfig(closest_marker, fill='red')
                self.coordList.delete(self.marker_dict[closest_marker[0]])
                label_str = self.getCoords(event, marker) + ' (maybe)'
                self.coordList.insert(self.marker_dict[closest_marker[0]], label_str)
        else:
            # Create dot
            marker = self.canvas.create_oval(x1, y1, x2, y2, fill='blue')
            label_str = self.getCoords(event, marker)
            self.coordList.insert(tk.END, label_str)

    def getCoords(self, event, marker):
        # Find coordinates in image and print to table
        self.marker_dict[marker] = self.coordList.size()
        canvas = event.widget
        x = canvas.canvasx(event.x)
        y = canvas.canvasy(event.y)
        label_str = '{},{}'.format(x,y)
        return label_str


    def save(self):
        # Get coordinates of all labels and write to file as metadata for image tile
        return None




app = Application()
app.master.title('Termite Mound Labeller')
app.mainloop()


