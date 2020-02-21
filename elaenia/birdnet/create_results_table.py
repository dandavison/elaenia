import sys
from collections import defaultdict
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd

from elaenia.birdnet import BirdnetResult
from elaenia.utils.matplotlib import float_to_rgb


HTML = """
<html>
  <head>
    <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
    <script>
      $(function () {
        $('[data-toggle="popover"]').popover({html: true, content: function() {return $("#popover-" + this.dataset.resultId).html()}, sanitize: false})
      })
    </script>
    <style>
      .table {
        overflow-y: auto;
      }
      .table table {
        border-collapse: collapse;
      }
      .table thead th {
        position: sticky;
        top: 0;
        background: #fff;
      }
      .popover {
        max-width: none;
        background: #eee;
      }
      body {
        margin: 20px;
      }
    </style>
  </head>
  <body>
    <h2>Automated identification of birds from audio recordings, by <a href="https://birdnet.cornell.edu/">BirdNET</a>.</h2>
    <ul>
      <li>Column 1 is a link to the original recording</li>
      <li>Column 2 is the true (primary) species in the recording, according to the recordist. Hover over this entry to see all the species identified by BirdNET in the recording, along with their times of appearance. These can be checked against the original recording.
      <li>Remaining columns show the total percentage confidence assigned by BirdNET to each species.
    </ul>
    %(table)s
  </body>
  %(popovers)s
</html>
"""

COLORMAP = plt.get_cmap("Reds")


button_with_popover = """
<button type="button" class="btn btn-lg btn-danger" data-result-id="%(resultid)s" data-trigger="hover" data-toggle="popover" title="All birds identified in %(recording)s" >%(text)s</button>
""".strip()


def create_results_table(paths):
    results = [BirdnetResult(path) for path in paths]
    df = _make_dataframe(results)
    io = StringIO()
    df.to_html(
        io,
        classes=["table"],
        escape=False,
        float_format=lambda f: f'<span style="color:{float_to_rgb(f/100, COLORMAP)}">%.0f</span>'
        % f,
    )
    table = io.getvalue()
    popovers = "\n".join(_get_popover(r) for r in results)
    with open("results.html", "w") as fp:
        fp.write(HTML % {"table": table, "popovers": popovers})


def _make_dataframe(results):
    all_species_to_percent = defaultdict(float)
    for r in results:
        for s, p in r.species_to_percent.items():
            all_species_to_percent[s] += p

    all_species = [
        s
        for _, s in sorted(
            zip(all_species_to_percent.values(), all_species_to_percent.keys()), reverse=True,
        )
    ]

    all_species = all_species[:15]

    row_factory = lambda: {s: 0.0 for s in all_species}
    rows = []
    for r in results:
        row = row_factory()
        row.update(r.species_to_percent)
        row["sort_key"] = r.true_species
        row["True species"] = button_with_popover.strip() % {
            "text": r.true_species,
            "recording": f"XC{r.id}",
            "resultid": r.id,
        }
        rows.append(row)

    columns = ["sort_key", "True species"] + all_species
    index = [f'<a href="{r.url}">XC{r.id}</a>' for r in results]
    df = pd.DataFrame.from_records(rows, index=index, columns=columns)
    df.sort_values("sort_key", inplace=True)
    del df["sort_key"]
    return df


def _get_popover(result, sort_by="confidence"):
    result.df["Confidence (%)"] = result.df["Confidence"] * 100
    result.df["Begin Time"] = minutes_and_seconds(result.df["Begin Time (s)"])
    result.df["End Time"] = minutes_and_seconds(result.df["End Time (s)"])
    result.df["sort_key"] = result.df["Confidence (%)"].round()
    if sort_by == "confidence":
        df = result.df.sort_values(["sort_key", "Begin Time (s)"], ascending=[False, True])
    elif sort_by == "time":
        df = result.df.sort_values(["Begin Time (s)"], ascending=[True])
    else:
        raise ValueError(f"Invalid `sort_by`: {sort_by}")
    table = (
        df[["Begin Time", "End Time", "Common Name", "Confidence (%)", "Rank"]]
        .to_html(float_format=lambda f: "%.0f" % f, index=False)
        .replace(r"\n", " ")
    )
    return f"""
    <div id="popover-{result.id}" style="display:none">
      {table}
    </div>
    """


def minutes_and_seconds(seconds):
    return ["%d:%02d" % divmod(s, 60) for s in seconds]


if __name__ == "__main__":
    paths = sys.argv[1:]
    create_results_table(paths)
