import re
import heapq

# BPE O(nlogn) encoder
def BPEncoder(inference_text, REGEX, mint_map, merge_rank, encodeMap):
  text = re.findall(REGEX, inference_text)
  encoded = []
  minHeap = []
  mint_map = {v:k for k,v in mint_map.items()} # pair -> token
  for word in range(len(text)):
    encoded.append([])
    for char in range(len(text[word])):
      encoded[word].append(encodeMap[text[word][char]])
      # saving pair locations
      if char<len(text[word])-1:
        pair = (encoded[word][char],encodeMap[text[word][char+1]])
        if pair in mint_map: minHeap.append((merge_rank[mint_map[pair]], pair, word, char))
  heapq.heapify(minHeap)
  # merge loop
  while minHeap:
    rank, pair, word, char = heapq.heappop(minHeap)

    if char+1>=len(encoded[word]): continue
    if (encoded[word][char], encoded[word][char+1]) != pair: continue

    else:
      # merging pair
      encoded[word][char:char+2] = [mint_map[pair]]
      # update left
      if char>0:
        pl = (encoded[word][char-1], mint_map[pair])
        if pl in mint_map: heapq.heappush(minHeap,(merge_rank[mint_map[pl]], pl, word, char-1))
      # update right
      if char<len(encoded[word])-2:
        pr = (mint_map[pair], encoded[word][char+1])
        if pr in mint_map: heapq.heappush(minHeap,(merge_rank[mint_map[pr]], pr, word, char))
  return encoded
