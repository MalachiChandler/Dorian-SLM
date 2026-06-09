from parameters import max_vocab_size, training_chunk_size, REGEX
import random
import heapq
import re
import pickle

##
with open('/content/drive/MyDrive/Colab Notebooks/DMT/50M_Wikipedia.txt', 'r') as Base:
  base = Base.read()
with open('/content/drive/MyDrive/Colab Notebooks/DMT/fine_tuning.txt', 'r') as Fine:
  fineTuning = Fine.read()
chars = sorted(list(set(base + fineTuning)))
##
# Base Encoder and Decoder
encodeMap = {}
decodeMap = {}
for i , char in enumerate(chars):
  encodeMap[char] = i
  decodeMap[i] = char
def old_encode(seq):
  return [encodeMap[c] for c in seq]
def old_decode(seq):
  seq = seq.tolist()
  return ''.join([decodeMap[i] for i in seq])
  
maxCode = len(list(set(fineTuning)))
chunk = get_chunk(fineTuning, training_chunk_size)
chunk = re.findall(REGEX, chunk)
mint_map = {}
merge_rank = {}
for char, token in encodeMap.items():
  merge_rank.update({token:0})
corpus = []
pair_counts = {}
pair_locations = {}

# Populating Data Structures
for word in range(len(chunk)):
  corpus.append([])
  for char in range(len(chunk[word])):
    corpus[word].append(encodeMap[chunk[word][char]])
    if char>0:
      # Counting occurences of each pair
      pair = (corpus[word][char-1],corpus[word][char])
      pair_counts[pair] = pair_counts.get(pair,0) + 1
      # saving location of the pair
      pair_locations.setdefault(pair, set()).add(word)
# putting occurences first for max heap
maxHeap = [(-v, k) for k, v in pair_counts.items()]
heapq.heapify(maxHeap)

print("Corpus length post REGEX:", len(corpus))
print("Sanity:", chunk[:20])
print(type(maxHeap))

### Training Tokenizer ###
startingV = maxCode
while maxCode < max_vocab_size:
  ## Sanity check
  if not maxHeap:
    if len(pair_counts) == 0: 
      print("Byte-Part-Encoding completed at:", maxCode, "tokens!")
      break
    else:
      print("Something went wrong, remaining pairs:", len(pair_counts))
  # Minting Loop
  heapMax = heapq.heappop(maxHeap)
  heapCount = -heapMax[0]
  pair = heapMax[1]

  if pair not in pair_counts: continue
  if heapCount != pair_counts[pair]:
    # lazy update
    heapq.heappush(maxHeap, (-pair_counts[pair], pair))
    continue
  else:
    maxCode += 1
    ## Merging in the corpus ##
    for word in list(pair_locations[pair]):
      char = 0
      beforePairCounts = {}
      replacement = []
      # counting pairs before & creating replacement with new token
      while char < len(corpus[word]):
        if char < len(corpus[word])-1:
          dupla = (corpus[word][char] , corpus[word][char+1])
          if dupla == pair:
            replacement.append(maxCode)
            beforePairCounts[dupla] = beforePairCounts.get(dupla,0) + 1
            char+=2
          else:
            beforePairCounts[dupla] = beforePairCounts.get(dupla,0) + 1
            replacement.append(corpus[word][char])
            char+=1
        else:
          replacement.append(corpus[word][char])
          char+=1
      corpus[word] = replacement
      # counting pairs after merge
      afterPairCounts = {}
      for char in range(len(replacement)):
        if char>0:
          dupla = (corpus[word][char-1],corpus[word][char])
          afterPairCounts[dupla] = afterPairCounts.get(dupla,0) + 1
      # Finding difference in before and after pair counts and updating lobal pair locations
      change = {}
      for dupla in set(beforePairCounts.keys()) | set(afterPairCounts.keys()):
        if dupla in beforePairCounts and dupla in afterPairCounts:
          change[dupla] = afterPairCounts[dupla] - beforePairCounts[dupla]
        elif dupla in beforePairCounts:
          change[dupla] = -beforePairCounts[dupla]
          if dupla in pair_locations:
            pair_locations[dupla].discard(word)
            if not pair_locations[dupla]: pair_locations.pop(dupla, None)
        elif dupla in afterPairCounts:
          change[dupla] = afterPairCounts[dupla]
          pair_locations.setdefault(dupla, set()).add(word)
        # updating global pair counts and priority queue
        pair_counts[dupla] = pair_counts.get(dupla,0) + change[dupla]
        if pair_counts[dupla] > 0:
          heapq.heappush(maxHeap, (-pair_counts[dupla], dupla))
        elif pair_counts[dupla] <= 0:
          pair_counts.pop(dupla, None)
    pair_locations.pop(pair, None)
    pair_counts.pop(pair, None)
  mint_map.update({maxCode : pair})
  merge_rank.update({ maxCode : max(merge_rank[pair[0]], merge_rank[pair[1]])+1 })
  decodeMap.update({maxCode : decodeMap[pair[0]] + decodeMap[pair[1]]})
  encodeMap.update({decodeMap[maxCode] : maxCode})
  # Sanity Checking
  if maxCode % round(max_vocab_size*0.10) == 0:
    print((maxCode/max_vocab_size)*100, "% complete, ", maxCode, " tokens", sep="")
## Checking and Saving
print(maxCode, startingV)
print(list(decodeMap.values())[29000:])
locations = [ '50kmint_map.pickle', '50kmerge_rank.pickle', '50kencodeMap.pickle', '50kdecodeMap.pickle' ]
importants = [ mint_map, merge_rank, encodeMap, decodeMap ]
for i in range(len(importants)):
  with open(locations[i], 'wb') as saver:
    pickle.dump(importants[i], saver, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved :)")
