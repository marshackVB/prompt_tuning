banking_77_prompt = """
You must classify questions from bank customer into one of the categories below.


Categories: 

Activate my card

Age limit question

Google Pay or Apple Pay support

ATMs support

Automatic top up

Balance not updated after bank transfer

Balance not updated after cheque or cash deposit

Beneficiary is not allowed

Cancel a transaction

Card is about to expire

Where are cards accepted

Delivery of new card or when will card arrive

Linking card to app

Card is not working

Card payment fee charged

Card payment wrong exchange rate

Card swallowed or not returned by ATM machine

Cash withdraw fee charged

Card payment not recognized

Change PIN number

Contactless payment not working

Countries where card is supported

Cash withdraw was declined

Transfer was declined

Edit personal details


Response:

User: {question}
AI:

"""


