library(readr)
library(dplyr)
library(DBI)
library("RPostgreSQL")

setwd("/home/selina/Desktop/DS/datascience/DPV/")
data0 <- read_delim(file = "BI_Raw_Data.csv", 
                    delim=";", 
                    col_names=TRUE, 
                    col_types=NULL, 
                    locale=locale(encoding="ISO-8859-1"))
# head(data0) # inspect first five rows

# Make table 'product'
product <- data0 %>%
  select(Product_Name, Product_Category) %>%
  rename(name = Product_Name, category = Product_Category) %>%
  arrange(name, category) %>%
  group_by(name, category) %>% # grouping the rows based on name and category
  distinct() %>% # retain only unique rows
  ungroup() %>% 
  mutate(productid = row_number()) # add identifier

# head(product)

# Make table 'customer'
customer <- data0 %>%
  select(Customer_Name, Customer_Country) %>%
  rename(name = Customer_Name, country=Customer_Country) %>%
  arrange(name, country) %>%
  group_by(name, country) %>%
  distinct() %>%
  ungroup() %>%
  mutate(customerid = row_number())

# head(customer)

# Make table 'sales'
sales <- data0 %>%
  select(Order_Date_Day, Product_Name, Product_Category, 
         Customer_Name, Customer_Country, Order_Price_Total)
sales <- sales %>%
  full_join(customer, by=c("Customer_Name"="name",
                          "Customer_Country"="country")) %>%
  select(-Customer_Name, -Customer_Country)
sales <- sales %>%
  full_join(product, by=c("Product_Name"="name",
                          "Product_Category"="category")) %>%
  select(-Product_Name, -Product_Category)

# Remove extra columns
sales <- sales[, !(colnames(sales) %in% c("salesid","category","country"))]

# head(sales)

drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, port=5432, host="bronto.ewi.utwente.nl",
                 dbname="dab_ds20211b_92", user="dab_ds20211b_92", 
                 password="esgL4S/HT7fgFST+", 
                 options="-c search_path=ass2")
dbWriteTable(con, "product", value = product, overwrite = T, row.names = F)
dbWriteTable(con, "customer", value = customer, overwrite = T, row.names = F)
dbWriteTable(con, "sales", value = sales, overwrite = T, row.names = F)

# Test to see if filled
# dbListTables(con)
# str(dbReadTable(con,"customer"))
# str(dbReadTable(con,"product"))

# Include the output of the following R-code:
dbGetQuery(con,
             "SELECT table_name FROM information_schema.tables
WHERE table_schema='ass2'") ## to get the tables from schema ass2
str(dbReadTable(con, c("ass2", "customer")))
str(dbReadTable(con, c("ass2", "product")))
str(dbReadTable(con, c("ass2", "sales")))
str(dbReadTable(con,"sales"))