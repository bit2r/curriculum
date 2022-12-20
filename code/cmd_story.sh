
curl https://raw.githubusercontent.com/jeroenjanssens/data-science-at-the-command-line/master/book/2e/data/ch02/movies.txt --output data/movies.txt

curl -sL "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt" |
sort | tee stopwords 

curl -sL "https://www.gutenberg.org/files/11/11-0.txt" |                        
tr '[:upper:]' '[:lower:]' |            
grep -oE "[a-z\']{2,}" |                
sort |              
grep -Fvwf stopwords |                  
uniq -c |           
sort -nr |          
head -n 10