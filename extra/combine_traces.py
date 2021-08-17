import json
from glob import glob

# directory containing the timeline-*.json partial traces
dir = '../log'

output = dict() # json object that will be written out
output['traceEvents'] = []
labels = set()
for trace in glob(f'{dir}/timeline-*.json'):
    with open(trace) as f:
        for node in json.loads(f.read())['traceEvents']:
            if 'dur' in node: # keep nodes that have a "dur"ation
                output['traceEvents'].append(node)
            elif node['name'] == 'process_name':
                labels.add(tuple(node))
            # ignore all others, I can't tell what they're for

# convert sets to lists for JSON printing
output['traceEvents'] = [list(a) for a in labels] + output['traceEvents']
with open('traceEvents.json', 'w') as f:
    f.write(json.dumps(output))

# To view the trace:
# - Go to "chrome://tracing" in Google Chrome or Chromium
# - Click load
# - Navigate to ./traceEvents.json
