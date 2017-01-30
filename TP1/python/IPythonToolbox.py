from IPython.display import HTML, display
import numpy as np

def printTable(table, title="", vHeader=[], hHeader=[], roundFloat=-1, prefix='', suffix=''):

    # Count rows and column
    rowCount = len(table)
    columnCount = 0
    for row in table:
        if columnCount < len(row):
            columnCount = len(row)

    if roundFloat>-1:
        for row in range(len(table)):
            for col in range(len(table[row])):
                if isinstance(table[row][col], float):
                    table[row][col] = round(table[row][col], roundFloat)

    tableData = [ [(prefix+str(table[row][col]))+suffix if col<len(table[row]) else "" for col in range(columnCount)] for row in range(rowCount)]

    html = "" if title == "" else "<caption>{}</caption>".format(title)
    if hHeader:
        if vHeader:
            html += '<td class="emptyCell" ></td>'
        for h in hHeader:
            html += ' <th class="hHeader">{}</th> '.format(str(h))
        for _ in range(columnCount-len(hHeader)):
            html += ' <th></th> '
        html = '<tr>{}</tr>'.format(html)

    for rowId, rowData in enumerate(tableData):
        line = ""
        if vHeader:
            if len(vHeader) > rowId:
                line = '<th class="vHeader">{}</th>'.format(str(vHeader[rowId]))
            else:
                line = '<th></th>'
        for d in rowData:
            line += ' <td>{}</td> '.format(d)
        html += '<tr>{}</tr>'.format(line)

    html = '<table class="custom">{}</table>'.format(html)
    display(HTML(html))