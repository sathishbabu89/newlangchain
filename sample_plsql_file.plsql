-- Sample PL/SQL Code to Insert Records in a Loop
DECLARE
  i NUMBER := 1;
BEGIN
  -- Loop to insert 10 records into a table
  FOR i IN 1..10 LOOP
    INSERT INTO employees (emp_id, emp_name, emp_salary)
    VALUES (i, 'Employee ' || i, 5000 + i * 100);
  END LOOP;

  COMMIT;
END;
