PyQL
====

The Pythonic Query Language.


The query syntax of the Pythonic Query Language (and descendant Domain Specific Query Languages) is:

fields @ conditions 
  fields is a field or a comma delimited list of fields. Like this: date, points, assists, rebounds. 
 conditions is a condition or an ` and ` delimited list of conditions. Like this: name = Lebron James and minutes > 30.

An example of valid PyQL from an NBA database of player performance is: 
 date, points, assists, rebounds @ name=Lebron James and minutes>30

Both fields and conditions are made up of terms. 
A term is a valid Python expression in a name space made up of: 
 database parameters 
 any imported python modules 
 PyQL Aggregators 
 other domain specific terms such as joins, constants, and Python methods.

