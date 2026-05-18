<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\Payment\StorePaymentRequest;
use App\Models\Payment;
use App\Services\PaymentService;
use Illuminate\Http\Request;

class PaymentController extends Controller
{
    public function __construct(private PaymentService $paymentService) {}
    /**
     * GET /admin/payments
     *
     * List payments. Filters: invoice_id, parent_id, student_id, method,
     * paid_from, paid_to
     */
    public function index(Request $request)
    {
        $query = Payment::with([
            'invoice.account.student.user',
            'guardian.user',
        ]);

        if ($request->filled('invoice_id')) {
            $query->where('invoice_id', $request->invoice_id);
        }

        if ($request->filled('parent_id')) {
            $query->where('parent_id', $request->parent_id);
        }

        if ($request->filled('student_id')) {
            $query->whereHas('invoice.account', function ($q) use ($request) {
                $q->where('student_id', $request->student_id);
            });
        }

        if ($request->filled('method')) {
            $query->where('method', $request->method);
        }

        if ($request->filled('paid_from')) {
            $query->where('paidat', '>=', $request->paid_from);
        }

        if ($request->filled('paid_to')) {
            $query->where('paidat', '<=', $request->paid_to);
        }

        $payments = $query->orderByDesc('paidat')
            ->paginate($request->input('per_page', 20));

        return response()->json($payments);
    }

    /**
     * POST /admin/payments
     *
     * Record a payment against an invoice. Transactionally:
     *   - validates amount ≤ outstanding on invoice
     *   - verifies the parent is a guardian of the invoiced student
     *   - updates invoice.status (unpaid | partial | paid)
     *   - decrements the student's fee account balance
     */
    public function store(StorePaymentRequest $request)
    {
        try {
            $payment = $this->paymentService->recordPayment($request->validated());
        } catch (\OverflowException $e) {
            return response()->json(json_decode($e->getMessage(), true), 422);
        } catch (\RuntimeException $e) {
            return response()->json(['message' => $e->getMessage()], 422);
        }

        return response()->json(
            $payment->load(['invoice.account.student.user', 'guardian.user']),
            201
        );
    }

    /**
     * GET /admin/payments/{id}
     */
    public function show(int $id)
    {
        $payment = Payment::where('payment_id', $id)
            ->with(['invoice.account.student.user', 'guardian.user'])
            ->firstOrFail();

        return response()->json($payment);
    }

    /**
     * DELETE /admin/payments/{id}
     *
     * Void a payment. Reverses the balance + invoice status changes.
     */
    public function destroy(int $id)
    {
        $this->paymentService->voidPayment($id);

        return response()->json(['message' => 'Payment voided successfully.']);
    }
}
