import random
import heapq
import re

# Getting random chunks from training text
def get_chunk(text, chunk_size):
  possible_chunks = round(len(text) / chunk_size)
  chunk_idx = random.randrange(possible_chunks)
  return text[chunk_idx*chunk_size : chunk_idx*chunk_size + chunk_size]

# Recursively finding the characters for a token
def findChars(token, mint_map, decodeMap, encodeMap):
  if token in decodeMap: return decodeMap[token]
  a = findChars(mint_map[token][0], mint_map, decodeMap, encodeMap)
  b = findChars(mint_map[token][1], mint_map, decodeMap, encodeMap)
  decodeMap.update({token:a+b})
  encodeMap.update({a+b:token})
  return a+b

# Getting strings for created tokens
def TokenStrings(mint_map, decodeMap, encodeMap):
  for token in mint_map.keys():
    findChars(token, mint_map, decodeMap, encodeMap)
  return decodeMap, encodeMap
