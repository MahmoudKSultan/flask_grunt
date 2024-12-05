
# encoding: utf-8
# coder:    Ishaq Khan
# version:  1.0

import os
import sys
import pymupdf
import csv

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

your_estimate_src = '1.pdf'
carrier_estimate_src = '2.pdf'

if not os.path.exists(your_estimate_src):
    print('Make sure "{}" file exists or change the path in the script and try again'.format(your_estimate_src))
    exit()

elif not os.path.exists(carrier_estimate_src):
    print('Make sure "{}" file exists or change the path in the script and try again'.format(carrier_estimate_src))
    exit()

o_and_p_items = 'ACOUSTICAL TREATMENTS', 'APPLIANCES', 'CABINETRY', 'CLEANING', 'GENERAL DEMOLITION', 'DOORS', 'DRYWALL', 'ELECTRICAL', 'FLOOR COVERING - CARPET', 'FLOOR COVERING - STONE', 'FLOOR COVERING - CERAMIC TILE', 'FINISH CARPENTRY / TRIMWORK', 'FINISH HARDWARE', 'FIREPLACES', 'FRAMING & ROUGH CARPENTRY', 'HAZARDOUS MATERIAL REMEDIATION', 'HEAT,  VENT & AIR CONDITIONING', 'INSULATION', 'LABOR ONLY', 'LIGHT FIXTURES', 'MARBLE - CULTURED OR NATURAL', 'MOISTURE PROTECTION', 'MIRRORS & SHOWER DOORS', 'PLUMBING', 'PAINTING', 'ROOFING', 'SCAFFOLDING', 'SIDING', 'TILE', 'TEMPORARY REPAIRS', 'USER DEFINED ITEMS', 'WINDOWS - ALUMINUM', 'WINDOWS - SLIDING PATIO DOORS', 'WINDOW TREATMENT', 'EXTERIOR STRUCTURES'

# --------------------------------------------------------------------

your_estimate_src_op = pymupdf.open(your_estimate_src)

for pg in your_estimate_src_op:
    if pg.search_for('Recap by Category with Depreciation'):
        tar_table_ind = pg.number

your_estimate_col_data = {}

for o_and_p_item in o_and_p_items:
    tab_pg_1 = your_estimate_src_op[tar_table_ind]
    tab_pg_2 = your_estimate_src_op[tar_table_ind + 1]

    for text_block in tab_pg_1.get_text('blocks') + tab_pg_2.get_text('blocks'):
        if text_block[4].startswith(o_and_p_item):
            vals = text_block[4].split('\n')
            your_estimate_col_data[vals[0]] = float(vals[1].replace(',', ''))

your_estimate_src_op.close()

# --------------------------------------------------------------------

carrier_estimate_src_op = pymupdf.open(carrier_estimate_src)

for pg in carrier_estimate_src_op:
    if pg.search_for('Recap by Category with Depreciation'):
        tar_table_ind = pg.number

carrier_estimate_col_data = {}

for o_and_p_item in o_and_p_items:
    tab_pg_1 = carrier_estimate_src_op[tar_table_ind]
    tab_pg_2 = carrier_estimate_src_op[tar_table_ind + 1]

    for text_block in tab_pg_1.get_text('blocks') + tab_pg_2.get_text('blocks'):
        if text_block[4].startswith(o_and_p_item):
            vals = text_block[4].split('\n')
            carrier_estimate_col_data[vals[0]] = float(vals[1].replace(',', ''))

carrier_estimate_src_op.close()

# --------------------------------------------------------------------

col_data = [['O&P item', 'Your estimate', 'Carrier estimate', 'Difference']]

for o_and_p_item in o_and_p_items:
    your_val = your_estimate_col_data.get(o_and_p_item, 'N/A')
    carrier_val = carrier_estimate_col_data.get(o_and_p_item, 'N/A')

    if 'N/A' in [your_val, carrier_val]:
        col_data.append([o_and_p_item, your_val, carrier_val, 'N/A'])
    else:
        col_data.append([o_and_p_item, your_val, carrier_val, carrier_val - your_val])

# --------------------------------------------------------------------

out_op = open('output.csv', newline='', mode='w', encoding='utf-8')
csv.writer(out_op).writerows(col_data)
out_op.close()

print('"output.csv" file generated successfully.')
