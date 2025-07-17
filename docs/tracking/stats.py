# # package = "polartoolkit"
# # Show overall downloads over time, excluding mirrors
# data = pypistats.overall(package, total=True, format="pandas")
# print(data)
# data = data.groupby("category").get_group("without_mirrors").sort_values("date")

# chart = data.plot(x="date", y="downloads", figsize=(10, 2))
# chart.figure.savefig("overall.png")

"""
Adapted from the package icepyx: https://github.com/icesat2py/icepyx

https://github.com/icesat2py/icepyx/blob/6c187bd35358d88083a5163d3491118aa1aad45c/doc/source/tracking/pypistats/get_pypi_stats.py
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import pypistats
import seaborn as sns

sns.set_theme()

cwd = pathlib.Path.cwd()

trackpath = f"{cwd}/docs/tracking/"
downloadfn = "downloads_data.csv"
sysdownloadfn = "sys_downloads_data.csv"

antarctic_plots_downloads = pypistats.overall(
    "antarctic_plots", total=True, format="pandas"
).drop(columns=["percent"])

polartoolkit_downloads = pypistats.overall(
    "polartoolkit", total=True, format="pandas"
).drop(columns=["percent"])

downloads = pd.concat([antarctic_plots_downloads, polartoolkit_downloads])

downloads = downloads[downloads.category != "Total"]

try:
    exist_downloads = pd.read_csv(trackpath + downloadfn)  # .drop(columns=['percent'])
    # exist_downloads = exist_downloads[exist_downloads.category != "Total"]
    dl_data = downloads.merge(
        exist_downloads, how="outer", on=["category", "date", "downloads"]
    )
except NameError:
    dl_data = downloads

total_downloads = dl_data.downloads.sum()

dl_data.sort_values(["category", "date"], ignore_index=True).to_csv(
    trackpath + downloadfn, index=False
)

dl_data = dl_data.groupby("category").get_group("without_mirrors").sort_values("date")

# sns.set(rc={'figure.figsize':(10,2)})
# fig = sns.lineplot(data=dl_data, x="date", y="downloads",)
# fig.set_xticks(range(len(dl_data.date)), )#labels=range(2011, 2019))
# fig.figure.savefig(trackpath + "downloads.png")


# fig, ax = plt.subplots(figsize=(10, 2))
# ax .plot(
#     dl_data.date,
#     dl_data.downloads,
#     label="Number of PyPI Downloads",
# )
# ax.set_xlabel('Date')
# ax.set_ylabel('Downloads')
# plt.savefig(trackpath + "downloads.png")


chart = dl_data.plot(
    x="date",
    y="downloads",
    figsize=(10, 2),
    legend=False,
)

chart.set_ylabel("Downloads")

chart.set_title(
    "PyPI Downloads of PolarToolkit",
    fontdict={"fontsize": "large"},
)

chart.text(
    0, 1, f"{total_downloads} total downloads", va="top", transform=plt.gca().transAxes
)

chart.figure.savefig(trackpath + "downloads.png")
