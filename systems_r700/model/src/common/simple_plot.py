'''
Copied over from Impinj_Lab_Tools simple plot module. 
Helper utilities for plotting x,y over Z and for doing contour plots
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import griddata
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


class SimplePlot(object):
    """
    Class for more easily creating basic plots with matplotlib.
    """
    def __init__(self, save=False, line_style="None"):
        self._save = save
        self._saved_figures = []
        self._line_style = line_style

    def simple_plot(self, x, y, label, title, xlabel, ylabel):
        plots = [{"x": x, "y": y, "label": label}]
        self.multi_plot(plots, title, xlabel, ylabel)

    def multi_plot(self, plots, title, xlabel, ylabel):
        fig = plt.figure(title)
        fig.clear()

        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
        ax.set_title(title)

        for plot in plots:
            ax.plot(plot["x"], plot["y"], label=plot["label"], marker=".", linestyle=self._line_style)

        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), prop={"size": 6})
        plt.show(block=False)

        self._save_figure(fig)

    def plot_x_y_columns_over_z(self, frame, x_col, y_col, z_col, title):
        plots = []
        unique_z_values = pd.unique(frame[z_col].ravel())

        for z_value in unique_z_values:
            temperature_plot = {
                "x": frame[frame[z_col] == z_value][x_col],
                "y": frame[frame[z_col] == z_value][y_col],
                "label": "{}={}".format(z_col, z_value)
            }

            plots.append(temperature_plot)

        self.multi_plot(plots, title, x_col, y_col)

    def plot_3d(self, x, y, z, label, title, xlabel, ylabel, zlabel):
        fig = plt.figure(title)
        fig.clear()
        ax = fig.gca(projection='3d')

        surf = ax.plot_trisurf(x, y, z, cmap=matplotlib.cm.get_cmap('jet'), linewidth=0)
        ax.set_zlim(np.amin(z), np.amax(z))
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show(block=False)

        self._save_figure(fig)

    def plot_contour(self, x, y, z, label, title, xlabel, ylabel, zlabel):
        fig = plt.figure(title)
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)

        X, Y, Z = self._grid(x, y, z)
        cs = ax.contour(X, Y, Z, 15, cmap=plt.cm.get_cmap('jet'))

        ax.clabel(cs, inline=1, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def _grid(self, x, y, z, res_x=100, res_y=100):
        xi = np.linspace(min(x), max(x), res_x)
        yi = np.linspace(min(y), max(y), res_y)
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method="cubic")
        return xi, yi, zi

    def show(self):
        plt.show()

    def _save_figure(self, figure):
        if self._save:
            self._saved_figures.append(figure)

    def save_all_to_pdf(self, output_path):
        if not self._save:
            raise Exception("Plots were not saved.")

        pdf = PdfPages(output_path)

        for figure in self._saved_figures:
            pdf.savefig(figure)

        pdf.close()
