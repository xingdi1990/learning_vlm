

# for FILE in output_pdfs/*; do 
#     echo $FILE; 
#     marker_single $FILE output_markdowns --batch_multiplier 2 --max_pages 10; 
#     done

# marker_single output_pdfs/0a83b7fd6a10b6880f097c3fe38d441c49f88a48.pdf \
#     output_markdowns --batch_multiplier 2 --max_pages 10


marker output_pdfs output_markdowns --workers 1