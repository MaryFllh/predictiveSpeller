import re
import numpy as np
import time
import itertools
import pickle

def uniqueOrdered(seq):
    seen = set()
    seen_add = seen.add # Faster than calculating the binding each time
    return [x for x in seq if not (x in seen or seen_add(x))]

def replaceWithDots(word, indexes):
	if len(indexes) == 0:
		return word

	return word[0:indexes[0]] + '.' + replaceWithDots(word, indexes[1:])[indexes[0]+1:]

def getPartialMatches(corpus, stub, num_misses, max_suggestions = 0):
	"""
	Returns a ranked list of words from the corpus, with up to num_misses substitutions
	from an open corpus (no context) of words.

	NOTE: by definition, this will return a superset of gPM(corpus, stub, num_misses-1, max_suggestions)
	"""
	comb = itertools.combinations(range(len(stub)), num_misses)
	word_list = []
	for pattern in comb:
		new_stub = replaceWithDots(stub, pattern)
		r = re.compile(new_stub)
		word_list = word_list + list(filter(r.match, corpus.keys()))

	word_list = list(set(word_list)) # Remove duplicates

	# Re-establish order based on corpus
	frequencies = [corpus[x] for x in word_list]
	ordered_words = [x for _,x in sorted(zip(frequencies, word_list), reverse=True)]
	word_frequencies = sorted(frequencies, reverse=True)
	
	if max_suggestions > 0:
		return (ordered_words[:max_suggestions], word_frequencies[:max_suggestions])
	else:
		return (ordered_words, ordered_frequencies)
	

def getNPartialMatches(corpus, stub, num_suggestions):
	"""
	Iteratively calls getPartialMatches until num_suggestions are found. Will
	only fail if the dictionary is too small.
	"""
	results = []
	distances = []
	for n in range(len(stub)+1):
		(partial_results, _) = getPartialMatches(corpus, stub, n, num_suggestions) # don't keep track of counts; we will consult the corpus ourselves
		results = uniqueOrdered(results + partial_results)
		while len(distances) < len(results):
			distances.append(n)

		if len(results) >= num_suggestions:
			results = results[:num_suggestions]
			distances = distances[:num_suggestions]
			break

	counts = [corpus[x] for x in results]
	# Normalization takes place elsewhere

	return (results, counts, distances)

def getNormalizedProbabilities(counts, distances, penalty = 0.8):
	"""
	Normalized probabilities normalizes counts into probabilities, penalizing elements at a higher distance
	(whatever that might mean). The penalty is the successive multiplication by the penalty term (default is
	0.8) to the count for any item.
	"""
	if not len(counts) == len(distances):
		raise RuntimeError('Parameter count mismatch. As usual.')

	# Penalize!	
	probs = [counts[x] * (penalty ** distances[x]) for x in range(len(counts))]

	# Normalize!
	psum = sum(probs)
	return [x/psum for x in probs]

def getStubLetterProbabilities(corpus, stub):
	"""
	Iterate through all 26 letters, calculate probabilities that each letter is next in the
	word based on the corpus. Returns a tuple of two sets:

	- the set if the most-recently-selected letter is accurate (stub == 'ban' => ..na, ..nb, etc.)
	- the set if the most recently-selected letter is not accurate* (stub == 'ban' => ...a, ...b, etc.)
	"""
	stub = stub.lower()
	correct_stub = '.' * (len(stub)-1) + stub[-1]
	incorrect_stub = '.' * len(stub)
	correct_probs = np.zeros(26)
	incorrect_probs = np.zeros(26)
	correct_sum = 0.
	incorrect_sum = 0.
	
	for letter in range(26):
		r = re.compile(correct_stub + chr(ord('a') + letter))
		correct_probs[letter] = len(list(filter(r.match, corpus)))
		correct_sum += correct_probs[letter]
		r = re.compile(incorrect_stub + chr(ord('a') + letter))
		incorrect_probs[letter] = len(list(filter(r.match, corpus)))
		incorrect_sum += incorrect_probs[letter]


	correct_probs /= correct_sum
	incorrect_probs /= incorrect_sum

	return (correct_probs, incorrect_probs)