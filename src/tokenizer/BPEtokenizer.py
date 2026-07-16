import random
import heapq
import re
import pickle
from itertools import islice
from .functions import get_chunk

### Tokenizer Class ###
class tokenizer():
  def __init__(self, encodeMap={}, decodeMap={}, mint_map={}, merge_rank={}, REGEX=""):
    self.encodeMap = encodeMap # str -> token
    self.decodeMap = decodeMap # token -> str
    self.mint_map = mint_map # token -> pair
    self.merge_rank = merge_rank # token -> rank
    self.REGEX = REGEX
    if encodeMap=={} and decodeMap=={} and mint_map=={} and merge_rank=={} and REGEX=="": print("Your tokenizer is empty!")

  def BPEncode(self,inference_text):
    text = re.findall(self.REGEX, inference_text)
    encoded = []
    minHeap = []
    mint_map_copy = {v:k for k,v in self.mint_map.items()} # pair -> token
    for word in range(len(text)):
      encoded.append([])
      for char in range(len(text[word])):
        encoded[word].append(self.encodeMap[text[word][char]])
        # saving pair locations
        if char<len(text[word])-1:
          pair = (encoded[word][char],self.encodeMap[text[word][char+1]])
          if pair in mint_map_copy: minHeap.append((self.merge_rank[mint_map_copy[pair]], pair, word, char))
    heapq.heapify(minHeap)
    # merge loop
    while minHeap:
      rank, pair, word, char = heapq.heappop(minHeap)

      if char+1>=len(encoded[word]): continue
      if (encoded[word][char], encoded[word][char+1]) != pair: continue

      else:
        # merging pair
        encoded[word][char:char+2] = [mint_map_copy[pair]]
        # update left
        if char>0:
          pl = (encoded[word][char-1], mint_map_copy[pair])
          if pl in mint_map_copy: heapq.heappush(minHeap,(self.merge_rank[mint_map_copy[pl]], pl, word, char-1))
        # update right
        if char<len(encoded[word])-2:
          pr = (mint_map_copy[pair], encoded[word][char+1])
          if pr in mint_map_copy: heapq.heappush(minHeap,(self.merge_rank[mint_map_copy[pr]], pr, word, char))
    return [token for word in encoded for token in word]

  def BPDecode(self, encoded_text):
    decoded = ""
    for token in encoded_text:
      decoded += self.decodeMap[token]
    return decoded

## Previously used training chunk size of 10_000_000 I think and max vocab size of 50000
  def train(self, max_vocab_size, training_chunk_size, training_data):
    encodeMap = self.encodeMap
    decodeMap = self.decodeMap
    mint_map = self.mint_map
    merge_rank = self.merge_rank
    REGEX = self.REGEX
    def get_chunk(text, chunk_size):
      possible_chunks = round(len(text) / chunk_size)
      chunk_idx = random.randrange(possible_chunks)
      return text[chunk_idx*chunk_size : chunk_idx*chunk_size + chunk_size]
    maxCode = len(encodeMap)
    chunk = get_chunk(training_data, training_chunk_size)
    chunk = re.findall(REGEX, chunk)
    if len(merge_rank) == 0:
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
    print("Sanity checks before training begins","\nIf corpus peek is all newlines something went wrong\nIf he type of the maxHeap is None something went wrong", "\nCorpus length post REGEX:", len(corpus), "\nCorpus peek:", chunk[:20], "\n", type(maxHeap), len(encodeMap), len(decodeMap), sep="")

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
    print(len(encodeMap), len(decodeMap))
    if len(encodeMap) == len(decodeMap) and len(encodeMap) == max_vocab_size:
      print("Tokenizer training complete! :)")
    else: print("Something isn't right. :(")
    self.encodeMap = encodeMap
    self.decodeMap = decodeMap
    self.mint_map = mint_map
    self.merge_rank = merge_rank
    return encodeMap, decodeMap, mint_map, merge_rank

  def pickle_save(self, locations):
    """
    Locations should be a list containing the locations to save pickle files for
    mint_map, merge_rank, encodeMap, decodeMap, and in that order.
    """
    importants = [ mint_map, merge_rank, encodeMap, decodeMap ]
    for i in range(len(importants)):
      with open(locations[i], 'wb') as saver:
        pickle.dump(importants[i], saver, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved :)")

  def pickle_load(self, locations):
    """
    Locations should be a list containing the locations to the pickle files for
    mint_map, merge_rank, encodeMap, decodeMap, and in that order.
    """
    importants = []
    for i in range(len(locations)):
      with open(locations[i], 'rb') as loader:
        importants.append(pickle.load(loader))
    self.mint_map = importants[0]
    self.merge_rank = importants[1]
    self.encodeMap = importants[2]
    self.decodeMap = importants[3]
    print(len(self.encodeMap), list(self.encodeMap.keys())[round(len(self.encodeMap)*0.95):])

  def reshape(self, desired_vocab_size):
    self.encodeMap = dict(islice(self.encodeMap.items(), desired_vocab_size))
    self.decodeMap = dict(islice(self.decodeMap.items(), desired_vocab_size))
    self.mint_map = {k:v for k,v in self.mint_map.items() if k in self.decodeMap}
    self.merge_rank = {k:v for k,v in self.merge_rank.items() if k in self.decodeMap}
    print("Complete! :)", len(self.encodeMap), len(self.decodeMap))
