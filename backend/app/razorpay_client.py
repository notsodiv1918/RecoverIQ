"""
The one real external integration in this project: creates an actual
Razorpay Payment Link via the test-mode API for the send_payment_link
action, instead of only simulating it.

If RAZORPAY_KEY_ID/SECRET aren't set, or the API call fails for any reason,
this falls back to a clearly-labeled SIMULATED link so the rest of the
pipeline is unaffected -- same silent-fallback pattern used for the
optional LLM diagnosis call in diagnosis.py. The fallback is labeled, not
hidden: audit trail and UI both show real=False when no real API call was
made, so this never quietly overstates what actually happened.

Get free test-mode keys: Razorpay Dashboard > Settings > API Keys (test
mode is the default toggle on signup, no live business verification needed).
Docs: https://razorpay.com/docs/api/payment-links/
"""
from . import config


def create_payment_link(txn) -> dict:
    """Returns {'real': bool, 'link_url': str, 'payment_link_id': str|None, 'note': str}."""
    if not config.USE_REAL_RAZORPAY:
        return {
            "real": False,
            "link_url": f"https://rzp.io/simulated/{txn.id}",
            "payment_link_id": None,
            "note": "RAZORPAY_KEY_ID/RAZORPAY_KEY_SECRET not set -- simulated link, no API call made.",
        }

    try:
        import razorpay
        client = razorpay.Client(auth=(config.RAZORPAY_KEY_ID, config.RAZORPAY_KEY_SECRET))

        customer_name = txn.customer.name if txn.customer else "Customer"
        customer_email = txn.customer.email if txn.customer else "customer@example.com"
        customer_phone = (txn.customer.phone if txn.customer else "9999999999")[:15]

        payload = {
            "amount": int(round(txn.amount * 100)),  # Razorpay amounts are in paise
            "currency": "INR",
            "description": f"Payment recovery -- transaction #{txn.id} ({txn.type})",
            "customer": {
                "name": customer_name,
                "email": customer_email,
                "contact": customer_phone,
            },
            "notify": {"sms": False, "email": False},  # our own guardrails control notification timing
            "reminder_enable": False,
            "notes": {
                "internal_transaction_id": str(txn.id),
                "root_cause": txn.root_cause or "unknown",
                "source": "ai-revenue-recovery-agent",
            },
        }
        link = client.payment_link.create(payload)
        return {
            "real": True,
            "link_url": link.get("short_url"),
            "payment_link_id": link.get("id"),
            "note": "Real Razorpay test-mode Payment Link created via API.",
        }
    except Exception as e:
        return {
            "real": False,
            "link_url": f"https://rzp.io/simulated/{txn.id}",
            "payment_link_id": None,
            "note": f"Razorpay API call failed ({type(e).__name__}: {e}), fell back to simulated link.",
        }