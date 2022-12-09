
files <- dir(path = "data/multicsv/", pattern = "*.csv", full.names =
               TRUE)
read_csv_quiet <- function(file) {
  read_csv(file, col_types = cols("T", "n", "f", "n", "n"), progress
           = FALSE) }
data <- files %>%
  # read_csv() on each file, reduce to one DF with rbind
  map(read_csv_quiet) %>%
  reduce(rbind)


# curl -O https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Ebook_Purchase_v1_00.tsv.gz 
# curl -O https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Ebook_Purchase_v1_01.tsv.gz
# 
# % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
# Dload  Upload   Total   Spent    Left  Speed
# 100 2565M  100 2565M    0     0  7518k      0  0:05:49  0:05:49 --:--:-- 8096k
# % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
# Dload  Upload   Total   Spent    Left  Speed
# 100 1234M  100 1234M    0     0  8020k      0  0:02:37  0:02:37 --:--:-- 7674k
# 
