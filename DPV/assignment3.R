library(readr)
library(dplyr)
library(DBI)
library("RPostgreSQL")
library(lubridate)

setwd("/home/selina/Desktop/DS/datascience/DPV/")

# Load data
data_main = read_delim(file = "SuperSales/SuperstoreSales_main.csv", 
                       delim=";", col_names=TRUE, col_types=NULL, 
                       locale=locale(encoding="ISO-8859-1"))

data_manager = read_delim(file = "SuperSales/SuperstoreSales_manager.csv", 
                       delim=";", col_names=TRUE, col_types=NULL, 
                       locale=locale(encoding="ISO-8859-1"))

data_returns = read_delim(file = "SuperSales/SuperstoreSales_returns.csv", 
                       delim=";", col_names=TRUE, col_types=NULL, 
                       locale=locale(encoding="ISO-8859-1"))

# Make table 'product'
product <- data_main %>%
  select("Product Name", "Product Category","Product Sub-Category" ) %>%
  rename(name="Product Name", category="Product Category", subcategory="Product Sub-Category") %>%
  arrange(name, category, subcategory) %>%
  group_by(name, category, subcategory) %>% # grouping the rows based on name and category
  distinct() %>% # retain only unique rows
  ungroup() %>% 
  mutate(productid = row_number()) # add identifier

# Make table 'customer'
customer <- data_main %>%
  select("Customer Name", "Province","Region", "Customer Segment" ) %>%
  rename(name="Customer Name", province="Province", region="Region", 
         segment="Customer Segment") %>%
  arrange(name, province, region, segment) %>%
  group_by(name, province, region, segment) %>% # grouping the rows based on name and category
  distinct() %>% # retain only unique rows
  ungroup() %>% 
  mutate(customerid = row_number()) # add identifier

# Make table 'returnstatus'
idReturnStatus <- c(0,1) 
returnstatus <- c("Returned", "Not Returned")  
ReturnStatus <- data.frame(idReturnStatus, returnstatus, stringsAsFactors=FALSE)

# Make table 'sales'
sales <- data_main %>%
  select("Order Date", Sales, "Order Quantity", "Unit Price", Profit, 
         "Shipping Cost", "Customer Name", "Order ID", "Ship Date", "Product Name") %>%
  rename(orderdate = "Order Date", sales = Sales, orderquantity = "Order Quantity", 
         unitprice = "Unit Price", profit = Profit, shippingcost = "Shipping Cost", 
         shipdate = "Ship Date") 

# Combine data to get product/customer id
sales <- sales %>%
  full_join(customer, by = c("Customer Name" = "name")) %>%
  select( -"Customer Name")
sales <- sales %>%
  full_join(product, by = c("Product Name" = "name")) %>%
  select( -"Product Name")
sales <- sales %>%
  full_join(data_returns, by = c("Order ID" = "Order ID")) %>%
  select( -"Order ID")

# Add returnstatus 
sales[c("Status")][is.na(sales[c("Status")])] <- "Not Returned"
sales <- sales %>%
  full_join(ReturnStatus, by = c("Status" = "returnstatus")) %>%
  select( -"Status")

# Determine the data types to see if change is needed
sapply(sales, class)

# Covert profit, shippingcost variables into numeric variables
sales$profit <- gsub(',', '.', sales$profit)
sales$shippingcost <- gsub(',', '.', sales$shippingcost)
sales$profit <- as.numeric(as.character(sales$profit), stringsAsFactors=FALSE)
sales$shippingcost <- as.numeric(as.character(sales$shippingcost), stringsAsFactors=FALSE)

# Convert character dates into numeric dates
sales$orderdate <- dmy(sales$orderdate)
sales$shipdate <- dmy(sales$shipdate)

# Check if the product is shipped late or not
sales$diff_in_days<- difftime(sales$shipdate ,sales$orderdate , units = c("days"))
sales$Late <- ifelse(sales$diff_in_days>=3, "late", "not late")

# Remove unwanted columns from sales table
sales <- sales[, !(colnames(sales) %in% c("shipdate","province", "region", "segment", 
                                          "diff_in_days", "category", "sub_category"))]

# Combine facts in sales table
sales <- sales %>%
  arrange(orderdate, customerid, productid, idReturnStatus, Late) %>%
  group_by(orderdate, customerid, productid, idReturnStatus, Late) %>%
  summarise(sales = sum(sales), orderquantity = sum(orderquantity), profit = sum(profit),
            shippingcost = sum(shippingcost)) %>%
  ungroup()


# Filling the data in PostgreSQL
drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, port=5432, host="bronto.ewi.utwente.nl",
                 dbname="dab_ds20211b_92", user="dab_ds20211b_92", 
                 password="esgL4S/HT7fgFST+", 
                 options="-c search_path=ass3")
dbWriteTable(con, "product", value = product, overwrite = T, row.names = F)
dbWriteTable(con, "customer", value = customer, overwrite = T, row.names = F)
dbWriteTable(con, "sales", value = sales, overwrite = T, row.names = F)
dbWriteTable(con, "ReturnStatus", value = ReturnStatus, overwrite = T, row.names = F)

# Output
dbGetQuery(con,
           "SELECT table_name FROM information_schema.tables
           WHERE table_schema='ass3'") 
str(dbReadTable(con, c("ass3", "customer")))
str(dbReadTable(con, c("ass3", "product")))
str(dbReadTable(con, c("ass3", "sales")))
str(dbReadTable(con, c("ass3", "ReturnStatus")))