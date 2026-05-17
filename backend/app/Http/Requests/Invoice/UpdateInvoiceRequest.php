<?php

namespace App\Http\Requests\Invoice;

use Illuminate\Foundation\Http\FormRequest;

class UpdateInvoiceRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'due_date'    => 'sometimes|date',
            'totalamount' => 'sometimes|numeric|min:0',
            'status'      => 'sometimes|in:unpaid,partial,paid,cancelled',
        ];
    }
}
