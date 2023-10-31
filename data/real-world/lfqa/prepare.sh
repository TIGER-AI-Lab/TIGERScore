cd ASQA && bash prepare.sh && cd ..
cd cosmos_qa && bash prepare.sh && cd ..
cd eli5 && bash prepare.sh && cd ..
cd FeTaQA && bash prepare.sh && cd ..
python aggregate.py
