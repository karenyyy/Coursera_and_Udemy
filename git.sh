git config credential.helper store
git add .
git commit -m 'update'
git push -u origin master

## to delete large files as well as cache on master
# git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch <file/dir>' HEAD
