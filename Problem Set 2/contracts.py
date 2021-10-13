import pandas as pd
import re
import sqlite3


def loadExcelToDB(excelfile, dbname):
    wb = pd.read_excel(excelfile, sheetname=None)
    action_table = pd.DataFrame()

    for name, sheet in wb.items():
        clean_name = re.sub('[(){}<>&]', '', name)
        dname = re.sub('[(){}<>1234567890 ]', '', name)
        dpk = re.sub('[a-zA-Z&(){}<> ]', '', name)
        if clean_name.endswith('00'):
            sheet['departmentID'] = dpk
            sheet['departmentName'] = dname
            action_table = action_table.append(sheet)

    vendorlist = action_table['Global Vendor Name'].unique()

    keys = [list(range(0,10))]
    vendordata = {'ID': [x for x in range(0,len(vendorlist))],'vendorName': vendorlist}
    vendor_table = pd.DataFrame(data=vendordata)
    vendor_table.set_index('ID', inplace=True)

    connection = sqlite3.connect(dbname)

    #clean up the database before re-writing the table
    connection.execute('''DROP TABLE IF EXISTS action_raw''')
    connection.execute('''DROP TABLE IF EXISTS vendor_raw''')
    connection.execute('''DROP TABLE IF EXISTS action_temp''')
    connection.execute('''DROP TABLE IF EXISTS actions''')
    connection.execute('''DROP TABLE IF EXISTS contractors''')

    action_table.to_sql(name='action_raw', con=connection)
    vendor_table.to_sql(name='vendor_raw', con=connection)

    connection.execute('''
        CREATE TABLE contractors (id INTEGER PRIMARY KEY NOT NULL, global_vendor_name VARCHAR);
    ''')

    connection.execute('''
        CREATE TABLE actions(id INTEGER NOT NULL,
                             actions INTEGER,
                             dollars REAL,
                             department VARCHAR,
                             contractor_id INTEGER,
                             FOREIGN KEY (contractor_id) REFERENCES contractors(id));
    ''')

    connection.execute('''
        INSERT INTO contractors (id,global_vendor_name)
        SELECT ID, vendorName
        FROM vendor_raw;
    ''')

    connection.execute('''
         CREATE TABLE action_temp as
         select action_raw.departmentid, action_raw.departmentName,action_raw.`Number of Actions` as actions, action_raw.`Dollars Obligated` as dollars, contractors.global_vendor_name, contractors.id from action_raw inner join contractors on 
         contractors.global_vendor_name = action_raw.`Global Vendor Name`;
    ''')

    connection.execute('''
        INSERT INTO actions (id,department,actions,dollars, contractor_id)
        SELECT departmentID, departmentName, actions, dollars, id
        FROM action_temp;	 
    ''')

    # Close connection
    connection.commit()
    connection.close()

    print ("Completed creating table \'contracts\' and \'actions\' in database \'contracts.db\'.")

def main():
    excelfile = 'Top_100_Contractors_Report_Fiscal_Year_2015.xls'
    dbname= 'contracts.db'
    loadExcelToDB(excelfile,dbname)

if __name__ == '__main__':
    main()
