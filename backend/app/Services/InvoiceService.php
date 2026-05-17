<?php

namespace App\Services;

use App\Models\Invoice;
use App\Models\Payment;
use Illuminate\Support\Facades\DB;

class InvoiceService
{
    /**
     * Record the remaining balance as a payment and mark the invoice as paid.
     *
     * Throws \RuntimeException if no guardian is linked to the student.
     */
    public function markPaid(Invoice $invoice, ?int $guardianId, ?string $method): void
    {
        DB::transaction(function () use ($invoice, $guardianId, $method) {
            $paidSum   = (float) $invoice->payments->sum('amount');
            $remaining = max(0, (float) $invoice->totalamount - $paidSum);

            if ($remaining > 0) {
                $parentId = $guardianId
                    ?? optional($invoice->account?->student?->guardians?->first())->parent_id;

                if (! $parentId) {
                    throw new \RuntimeException(
                        'Cannot record payment: no guardian linked to this student. Please link a parent first.'
                    );
                }

                Payment::create([
                    'invoice_id' => $invoice->invoice_id,
                    'parent_id'  => $parentId,
                    'amount'     => $remaining,
                    'method'     => $method ?? 'cash',
                    'paidat'     => now(),
                    'status'     => 'completed',
                ]);
            }

            $invoice->update(['status' => 'paid']);

            if ($invoice->account) {
                $invoice->account->update([
                    'paid_amount' => $invoice->totalamount,
                    'balance'     => 0,
                    'status'      => 'paid',
                ]);
            }
        });
    }

    /**
     * Delete an invoice, refusing if payments have already been recorded.
     *
     * Throws \RuntimeException if the invoice has payments.
     */
    public function deleteInvoice(Invoice $invoice): void
    {
        if ($invoice->payments()->exists()) {
            throw new \RuntimeException(
                'Cannot delete this invoice: payments have been recorded against it.'
            );
        }

        $invoice->delete();
    }
}
