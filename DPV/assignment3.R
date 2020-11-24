library(readr)
library(dplyr)
library(DBI)
library("RPostgreSQL")

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

