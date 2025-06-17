from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunking(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)


def is_overlap(word_bbox, table_bbox):
    wx0, wtop, wx1, wbot = word_bbox
    tx0, ttop, tx1, tbot = table_bbox

    horizontal_overlap = wx1 > tx0 and wx0 < tx1
    vertical_overlap = wbot > ttop and wtop < tbot
    return horizontal_overlap and vertical_overlap


def check_row(row):
    return [cell if cell is not None else " " for cell in row]


def table_to_markdown(table):
    if not table or not table[0]:
        return ""

    header = " | ".join(check_row(table[0]))
    separator = " | ".join(["---"] * len(table[0]))
    rows = [" | ".join(check_row(row)) for row in table[1:]]
    
    return "\n".join([header, separator] + rows)
