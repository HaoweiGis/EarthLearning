去除方式：
sed -i 's/^M//g' filename
sed -i 's/^[//g' filename
sed $'s/[^[:print:]\t]//g' file.txt

while read line;do cp ../deeps/images/$line images/;done < all.txt


