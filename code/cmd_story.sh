curl -s "https://raw.githubusercontent.com/statkclee/ml/gh-pages/data/B%EC%82%AC%EA%B0%90%EA%B3%BC_%EB%9F%AC%EB%B8%8C%EB%A0%88%ED%84%B0.txt" | \
grep -oE '\w+' | \
sort | \
uniq -c | \
sort -nr | \
head -n 5
