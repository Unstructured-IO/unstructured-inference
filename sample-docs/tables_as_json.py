# the names of the data in each cell
_schema = [
    "cell text",
    "row_nums",
    "column_nums"
]

# cleaner, well, at least leaner, format of cell data; maybe we should also consider using this for
# the code in tables.py
_data = {
	"table-multi-row-column-cells.png": [
		("Disability\nCategory", [0, 1], [0]),
		("Participants", [0, 1], [1]),
		("Ballots\nCompleted", [0, 1], [2]),
		("Ballots\nIncomplete/\nTerminated", [0, 1], [3]),
		("Results", [0], [4, 5]),
		("Accuracy", [1], [4]),
		("Time to\ncomplete", [1], [5]),
		("Blind", [2], [0]),
		("Low Vision", [3], [0]),
		("Dexterity", [4], [0]),
		("Mobility", [5], [0]),
		("5", [2], [1]),
		("5", [3], [1]),
		("5", [4], [1]),
		("3", [5], [1]),
		("1", [2], [2]),
		("2", [3], [2]),
		("4", [4], [2]),
		("3", [5], [2]),
		("4", [2], [3]),
		("3", [3], [3]),
		("1", [4], [3]),
		("0", [5], [3]),
		("34.5%, n=1", [2], [4]),
		("98.3% n=2\n(97.7%, n=3)", [3], [4]),
		("98.3%, n=4", [4], [4]),
		("95.4%, n=3", [5], [4]),
		("1199 sec, n=1", [2], [5]),
		("1716 sec, n=3\n(1934 sec, n=2)", [3], [5]),
		("1672.1 sec, n=4", [4], [5]),
		("1416 sec, n=3", [5], [5])
	]
}

# actual cell data in the same format as those returned by tables.py
data = {fname: [dict(zip(_schema, cell)) for fname, cell in _data.items()]}
