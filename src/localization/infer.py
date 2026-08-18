import argparse, json
from .inference import localize
parser = argparse.ArgumentParser(description='Run classical SEM localization.')
parser.add_argument('--search', required=True); parser.add_argument('--reference', required=True)
args = parser.parse_args()
print(json.dumps(localize(args.search, args.reference), indent=2))
