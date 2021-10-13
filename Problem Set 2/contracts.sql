    CREATE TABLE contractors (id INTEGER PRIMARY KEY NOT NULL, global_vendor_name VARCHAR);

	CREATE TABLE actions(id INTEGER NOT NULL,
						 actions INTEGER,
						 dollars REAL,
						 department VARCHAR,
						 contractor_id INTEGER,
						 FOREIGN KEY (contractor_id) REFERENCES contractors(id));
	
	
    INSERT INTO contractors (id,global_vendor_name)
	SELECT ID, vendorName
	FROM vendor_raw;
	
	 CREATE TABLE action_temp as
	 select action_raw.departmentid, action_raw.departmentName,action_raw.`Number of Actions` as actions, action_raw.`Dollars Obligated` as dollars, contractors.global_vendor_name, contractors.id from action_raw inner join contractors on 
	 contractors.global_vendor_name = action_raw.`Global Vendor Name`;
	 
	INSERT INTO actions (id,department,actions,dollars, contractor_id)
	SELECT departmentID, departmentName, actions, dollars, id
	FROM action_temp;
	 