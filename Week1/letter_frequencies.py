"""
This script is used to output the relative frequency of letters from all words in an online dictionary
"""

import matplotlib.pyplot as plt


def process_words():
    """
    This method reads every word in the dictionary and returns histogram of letter frequencies
    """
    file = open("MIT_Dictionary.txt", "r")
    # making a list of words in the dictionary
    word_list = [line[0:-2] for line in file if line[0:-2] != '']
    file.close()  # closing our file

    # pulling dictionary of respective letter occurrences. K:V --> letter:occurrences
    letter_occurs = calc_letters(word_list)

    # pulling total number of letters analyzed
    total_letter_num = sum(letter_occurs.values())

    # plotting letter vs occurrence of that letter from the letter_occurs dictionary

    # processing our x array
    letters_arr = [i for i in range(len(letter_occurs.keys()))]  # making faux array of letters [0-->25]
    xlabels = list(letter_occurs.keys())  # making array of our actual labels (list of letter keys from dictionary)
    plt.xticks(letters_arr, xlabels)  # overlaying our actual x labels (letters) on the x axis
    # creating array of letter frequencies
    letters_freq = [occurrence / total_letter_num for occurrence in letter_occurs.values()]
    # generating plot and showing
    bars = plt.bar(letters_arr, height=letters_freq, width=0.7)
    plt.title('Respective Frequencies of Letters from All Words in Dictionary')
    plt.xlabel('Letter')
    plt.ylabel('Respective Letter Frequencies')

    # labeling each bar w/ its respective frequency:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .001, str(round(yval, 3)), horizontalalignment="center")
    # showing plot
    plt.show()

    # printing highest/lowest frequency letter to console:
    max_freq, most_freq_letter = max(letters_freq), None
    min_feq, least_frq_letter = min(letters_freq), None
    for letter, freq in zip(letter_occurs.keys(), letters_freq):
        if freq == max_freq:
            print(f'The letter {letter} has the max frequency of {round(freq * 100, 3)}%')
            most_freq_letter = letter
        elif freq == min_feq:
            print(f'The letter {letter} has the min frequency of {round(freq * 100, 3)}%')
            least_frq_letter = letter
    print(
        f'The most frequent letter {most_freq_letter} was {round(max_freq / min_feq, 3)} '
        f'times more represented that the least frequent letter {least_frq_letter}')


def calc_letters(word_list):
    """
    This function accepts an array of words and returns the occurances of letters in that array
    Letter Frequencies are maintained in a dictionary of letter:occurances
    """
    # appending init counts to our dictionary
    letters_dict = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz'}
    # iterating over each word in the word list
    for curr_word in word_list:
        # iterating over every char in the passed letter
        for char in curr_word:
            letters_dict[char] += 1
    return letters_dict


if __name__ == "__main__":
    process_words()
