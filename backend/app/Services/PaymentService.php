<?php

namespace App\Services;

use App\Models\Invoice;
use App\Models\Payment;
use Illuminate\Support\Facades\DB;

class PaymentService
{
    /**
     * Record a payment against an invoice inside a transaction.
     *
     * Throws \RuntimeException for any business rule violation so the
     * controller can format the appropriate HTTP response.
     */
    public function recordPayment(array $data): Payment
    {
        return DB::transaction(function () use ($data) {
            $invoice = Invoice::where('invoice_id', $data['invoice_id'])
                ->lockForUpdate()
                ->with('account')
                ->firstOrFail();

            if ($invoice->status === 'cancelled') {
                throw new \RuntimeException('Cannot record a payment against a cancelled invoice.');
            }

            $isGuardian = DB::table('studentguardian')
                ->where('parent_id', $data['parent_id'])
                ->where('student_id', $invoice->account->student_id)
                ->exists();

            if (! $isGuardian) {
                throw new \RuntimeException('This parent is not a guardian of the invoiced student.');
            }

            $paidTotal   = (float) $invoice->payments()->sum('amount');
            $outstanding = (float) $invoice->totalamount - $paidTotal;

            if ((float) $data['amount'] > $outstanding + 0.00001) {
                throw new \OverflowException(
                    json_encode([
                        'message'     => 'Payment amount exceeds the invoice outstanding balance.',
                        'outstanding' => round($outstanding, 2),
                    ])
                );
            }

            $payment = Payment::create([
                'invoice_id' => $data['invoice_id'],
                'parent_id'  => $data['parent_id'],
                'amount'     => $data['amount'],
                'method'     => $data['method'],
                'paidat'     => $data['paidat'] ?? now(),
            ]);

            $this->syncInvoiceStatus($invoice, $paidTotal + (float) $data['amount']);
            $this->decrementAccountBalance($invoice->account, (float) $data['amount']);

            return $payment;
        });
    }

    /**
     * Void a payment and reverse its effect on the invoice and account balance.
     */
    public function voidPayment(int $id): void
    {
        DB::transaction(function () use ($id) {
            $payment = Payment::where('payment_id', $id)->lockForUpdate()->firstOrFail();

            $invoice = Invoice::where('invoice_id', $payment->invoice_id)
                ->lockForUpdate()
                ->with('account')
                ->firstOrFail();

            $amount = (float) $payment->amount;

            $payment->delete();

            $remainingPaid = (float) $invoice->payments()->sum('amount');
            $this->syncInvoiceStatus($invoice, $remainingPaid);

            $account = $invoice->account;
            $account->balance = (float) $account->balance + $amount;
            $account->save();
        });
    }

    private function syncInvoiceStatus(Invoice $invoice, float $paidTotal): void
    {
        $total = (float) $invoice->totalamount;

        if ($paidTotal + 0.00001 >= $total) {
            $invoice->status = 'paid';
        } elseif ($paidTotal > 0) {
            $invoice->status = 'partial';
        } else {
            $invoice->status = 'unpaid';
        }

        $invoice->save();
    }

    private function decrementAccountBalance($account, float $amount): void
    {
        $account->balance = max(0, (float) $account->balance - $amount);
        $account->save();
    }
}
