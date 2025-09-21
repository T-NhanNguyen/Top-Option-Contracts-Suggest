def print_beaver(message=""):
    print(fr'''
     .-"""-.__     {message}
    /      ' o'\
 ,-;  '.  :   _c
:_."\._ ) ::-"
       ""m "m
    ''')

def print_wizard(message=""):
    print(r'''
                      ____
                     / ___`\
         /|         ( (   \ \
    |^v^v  V|        \ \/) ) )
    \  ____ /         \_/ / /
    ,Y`    `,            / /
    ||  -  -)           { }
    \\   _\ |           | |
     \\ / _`\_         / /
     / |  ~ | ``\     _|_|
  ,-`  \    |  \ \  ,//(_}
 /      |   |   | \/  \| |
|       |   |   | '   ,\ \
|     | \   /  /\  _/`  | |
\     |  | |   | ``     | |
 |    \  \ |   |        | |
 |    |   |/   |        / /
 |    |        |        | |
    ''')

def print_wizard_message(message=""):
    print(fr'''
                  .
                   .
         /^\     .
    /\   "V"
   /__\   I      O  o
  //..\\\\  I     .
  \].`[/  I
  /l\/j\  (]    .  O
 /. ~~ ,\/I          .
 \\\\L__j^\/I       o
  \/--v]  I     o   .
  |    |  I   _________
  |    |  I c(`       ')o
  |    l  I   \.     ,/
_/j  L l\_!  _//^---^\\\\_    {message}
    ''')
def print_wizard(message=""):
    print(r'''
                             /\
                            /  \
                           |    |
                         --:"""":--
                           :'_' :
                           _:"":\___
            ' '      ____.' :::     '._
           . *=====<<=)           \    :
            .  '      '-'-'\_      /'._.'
                             \====:_ ""
                            .'     \\
                           :       :
                          /   :    \
                         :   .      '.
         ,. _            :  : :      :
      '-'    ).          :__:-:__.;--'
    (        '  )        '-'   '-'
 ( -   .00.   - _
(    .'  _ )     )
'-  ()_.\,\,   -'
    ''')
if __name__ == "__main__":
   print_wizard(r'''Cache Statistics:
{'stock_info_cache_size': 1, 'option_chain_cache_size': 16, 'expiration_cache_size': 1}''')