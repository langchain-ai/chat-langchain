# Run this script from AWS CloudShell
# > nano update_s3_metadata.sh
# ( Copy and paste the script ) 
# > chmod +x update_s3_metadata.sh
# > ./update_s3_metadata.sh > log_update_s3_metadata.txt 2>&1

#!/bin/bash

BUCKET_NAME="croptalk-spoi"

# List and update PDF files
aws s3 ls s3://$BUCKET_NAME/ --recursive | grep '.pdf' | awk '{print $4}' | while read -r file_key; do
    echo "Updating file: $file_key"
    aws s3api copy-object \
        --bucket $BUCKET_NAME \
        --copy-source $BUCKET_NAME/$file_key \
        --key $file_key \
        --metadata-directive REPLACE \
        --content-disposition "inline; filename='$(basename "$file_key")'" \
        --content-type application/pdf \
        --acl public-read
    sleep 1
done
