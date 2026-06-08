# BPE O(n) decoder
def BPDecoder(encoded_text, decodeMap):
  decoded = ""
  for word in range(len(encoded_text)):
    for char in range(len(encoded_text[word])):
      decoded += decodeMap[encoded_text[word][char]]
  return decoded
