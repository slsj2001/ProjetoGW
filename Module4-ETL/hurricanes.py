from bs4 import BeautifulSoup
import re
import sqlite3

def parse_table(table):
    pcol = []
    for row_i, row in enumerate(table.find_all('tr')):
        row_dict = {}
        if row_i == 0:
            th_tags = row.find_all('th')
            if (len(th_tags) > 0 and len(pcol) == 0):
                for th in th_tags:
                    col_heading = th.get_text(separator=' ').rstrip('\n').lower()
                    col_heading = re.sub(' +', ' ', col_heading)
                    pcol.append(col_heading)
        else:
            td_tags = row.find_all('td')
            if len(td_tags) > 0 and len(pcol) > 0:
                for col_index, col_value in enumerate(td_tags):
                    if pcol[col_index] in numcol:
                        try:
                            row_dict[pcol[col_index]] = float(col_value.text)
                        except ValueError:
                            item = col_value.text.rstrip('\n').lower()
                            if item in ['not known', 'unknown', 'Unknown']:
                                item = None
                            elif item == 'none':
                                item = 0
                            else:
                                pattern = re.compile(r'(,|\+|\$| |~|>|≥)')
                                item = pattern.sub('', item)
                                item = item.replace(u'\xa0', '')
                                if 'million' in item:
                                    item = item.replace('million', '')
                                    item = float(item)
                                    item = item * 1000000
                                elif 'billion' in item:
                                    item = item.replace('billion', '')
                                    item = float(item)
                                    item = item * 1000000000
                                try:
                                    item = float(item)
                                except:
                                    pass
                            row_dict[pcol[col_index]] = item
                    else:
                        pattern = re.compile(r'("|\')')
                        row_dict[pcol[col_index]] = col_value.get_text(separator=' ').rstrip('\n')
                    data_dict[int(row_dict[pcol[0]])] = row_dict


def createdb(db_name):
    connection = sqlite3.connect(db_name)
    connection.execute('''DROP TABLE IF EXISTS atlantic_hurricanes''')
    c = connection.cursor()
    c.execute('''CREATE TABLE atlantic_hurricanes
    (year INT PRIMARY KEY NOT NULL,
    tropical_storms INT,
    hurricanes INT,
    major_hurricanes INT,
    deaths INT,
    damage REAL,
    notes TEXT)''')
    for yr in data_dict:
        year = data_dict[yr]['year']
        tropical_storms = data_dict[yr]['number of tropical storms']
        hurricanes = data_dict[yr]['number of hurricanes']
        major_hurricanes = data_dict[yr]['number of major hurricanes']
        deaths = data_dict[yr]['deaths']
        if 'damage usd' in data_dict[yr]:
            damage = data_dict[yr]['damage usd']
        else:
            damage = None
        if 'notes' in data_dict[yr]:
            notes = ("Notes: " + data_dict[yr]['notes'])
            if 'strongest storm' in data_dict[yr]:
                notes += ("\nStrongest Storm: " +
                          data_dict[yr]['strongest storm'])
            elif 'retired  names' in data_dict[yr]:
                notes += ("\nRetired Names: " +
                          data_dict[yr]['retired  names'])
        else:
            notes = ''
        with connection:
            c.execute('''INSERT INTO Atlantic_hurricanes 
            VALUES(:year,
            :tropical_storms,
            :hurricanes,
            :major_hurricanes,
            :deaths,
            :damage,
            :notes)''', {
                'year': year,
                'tropical_storms': tropical_storms,
                'hurricanes': hurricanes,
                'major_hurricanes': major_hurricanes,
                'deaths': deaths,
                'damage': damage,
                'notes': notes
            })


data_dict = {}
numcol = ['year', 'number of tropical storms', 'number of hurricanes', 'number of major hurricanes','deaths', 'damage usd']

htmlfile = open('hurricanes.html', encoding="utf8")
soup = BeautifulSoup(htmlfile, 'html.parser')
tables = soup.find_all('table', class_='wikitable sortable')
for table in tables:
    parse_table(table)

db_file_name = 'hurricanes.db'
createdb(db_file_name)
