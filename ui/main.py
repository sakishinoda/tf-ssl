import tkinter as tk
from PIL import Image, ImageTk

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid(padx=10, pady=10)
        self.createWidgets()
        self.showImage()
        self.marker_dict = dict()
        self.currentMode = 0

    def createWidgets(self):

        self.canvas = tk.Canvas(self, width=256, height=256, bd=5)
        self.canvas.grid(row=0, column=1, columnspan=3)

        self.canvas.bind('<Button-1>', self.editLabel)

        # Mode buttons
        self.modes = tk.LabelFrame(self, text='Mode', padx=5, pady=5)
        self.modeButtons = []
        for i, mode in enumerate(['Label', 'Delete', 'Maybe']):
            self.modeButtons[i] = tk.Button(self.modes, text=mode, command=self.activateMode(i))
        self.modes.grid(row=0, column=2)

        # # Buttons
        # self.deleteButton = tk.Button(self, text='Delete Mode', command=self.deleteMode)
        # self.deleteButton.grid(row=1, column=1)
        # self.saveButton = tk.Button(self, text='Save')
        # self.saveButton.grid(row=1, column=2)
        # self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        # self.quitButton.grid(row=1, column=3)

        # # Set up label listbox
        # self.labelBox = tk.Listbox(self)
        # self.labelBox.grid(row=2, columnspan=3)


    def showImage(self):
        photo = ImageTk.PhotoImage(Image.open('sample.jpeg'))
        self.canvas.create_image(0, 0, anchor='nw', image=photo)
        self.canvas.current_image = photo

    def activateMode(self, mode):
        for button in self.modeButtons:
            button.config(relief=tk.RAISED)
        self.modeButtons[mode].config(relief=tk.SUNKEN)
        self.currentMode = mode


    def editLabel(self, event):
        # marker = self.canvas.gettags(tk.CURRENT)
        self.canvas.itemconfig(tk.CURRENT, fill='red')

        # self.labelBox.delete(self.marker_dict[marker[0]])
        # self.labelBox.insert(self.marker_dict[marker[0]], '({},{}) MAYBE'.format(x, y))
        self.editMenu.post(event.x_root, event.y_root)

    def addLabel(self, event):
        # Specify size of marker
        rad = 5

        x1, y1 = (event.x - rad), (event.y - rad)
        x2, y2 = (event.x + rad), (event.y + rad)

        # Create horizontal line
        # self.canvas.create_line(x1, event.y, x2, event.y, fill='blue')
        # Create vertical line
        # self.canvas.create_line(event.x, y1, event.x, y2, fill='blue')

        ## Create dot
        marker_tag = self.canvas.create_oval(x1, y1, x2, y2, fill='blue')
        self.marker_dict[marker_tag] = self.labelBox.size()
        print(self.marker_dict)

        # Find coordinates in image and print to table
        canvas = event.widget
        x = canvas.canvasx(event.x)
        y = canvas.canvasy(event.y)
        self.labelBox.insert(tk.END, '({},{})'.format(x,y))


    def save(self):
        # Get coordinates of all labels and write to file as metadata for image tile
        return None




app = Application()
app.master.title('Termite Mound Labeller')
app.mainloop()


